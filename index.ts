/**
 * Routera provider extension for pi.
 *
 * Routera exposes two compatible wire formats behind one account:
 *
 *   - OpenAI-compatible models: https://api.routera.one/v1
 *   - Anthropic-compatible Claude models: https://api.routera.one
 *
 * The model catalog is refreshed from both compatibility-specific /models
 * endpoints. Anthropic models use pi's Anthropic Messages API and every other
 * model uses pi's OpenAI Chat Completions API.
 *
 * Auth: `/login routera` prompts for a Routera API key
 * (https://www.routera.one/account?tab=apiKeys), with ROUTERA_API_KEY as an
 * automatic fallback.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	anthropicMessagesApi,
	createProvider,
	envApiKeyAuth,
	openAICompletionsApi,
	type ApiKeyAuth,
	type Model,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai/compat";

const PROVIDER_ID = "routera";
const ROUTERA_BASE_URL = "https://api.routera.one";
const OPENAI_BASE_URL = `${ROUTERA_BASE_URL}/v1`;
const OPENAI_MODELS_URL = `${OPENAI_BASE_URL}/models`;
const ANTHROPIC_MODELS_URL = `${OPENAI_BASE_URL}/models?compat=anthropic`;

const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;
// OpenAI's GPT-5-class reasoning models allow 128k output; reasoning tokens
// count against the completion budget, so keep the full headroom.
const OPENAI_REASONING_MAX_TOKENS = 128_000;
// Anthropic's synchronous Messages API allows 128k output for 1M-context
// Claude models and 64k for current 200k-context Claude models. The 300k
// batch-only beta is intentionally not used for interactive pi requests.
const ANTHROPIC_1M_CONTEXT_MAX_TOKENS = 128_000;
const ANTHROPIC_DEFAULT_MAX_TOKENS = 64_000;
const DISCOVERY_TIMEOUT_MS = 15_000;

/**
 * Routera documents `Authorization: Bearer <key>` as its canonical auth
 * header. pi's Anthropic client adds `x-api-key` for its models and OpenAI's
 * client adds `Authorization` itself, but declaring the Bearer header on the
 * resolved auth guarantees it is present for every Routera request — both
 * endpoints accept it — regardless of how each client assembles headers.
 */
function bearerHeaderAuth(inner: ApiKeyAuth): ApiKeyAuth {
	return {
		name: inner.name,
		login: inner.login,
		async resolve(input) {
			const result = await inner.resolve(input);
			if (!result?.auth.apiKey) return result;
			return {
				...result,
				auth: {
					...result.auth,
					headers: { ...result.auth.headers, Authorization: `Bearer ${result.auth.apiKey}` },
				},
			};
		},
	};
}

type RouteraApi = "anthropic-messages" | "openai-completions";

interface RouteraModelRecord {
	id: string;
	name?: string;
	owned_by?: string;
	context_length?: number;
	max_output_tokens?: number | null;
	max_tokens?: number | null;
	architecture?: {
		modality?: string;
	} | null;
	pricing?: {
		prompt?: string | null;
		completion?: string | null;
	} | null;
}

interface RouteraModelsResponse {
	data?: RouteraModelRecord[];
}

function isPositiveNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * Routera's stable ids are vendor-prefixed, but the ownership field is the
 * stronger signal when it is present. The name fallbacks cover unprefixed
 * Claude aliases returned by some catalog versions.
 */
export function isAnthropicModel(model: Pick<RouteraModelRecord, "id" | "owned_by">): boolean {
	const owner = model.owned_by?.toLowerCase() ?? "";
	const id = model.id.toLowerCase();
	return (
		owner === "anthropic" ||
		owner.startsWith("anthropic/") ||
		id.startsWith("anthropic/") ||
		/(^|[/_.-])(claude|opus|sonnet|haiku|fable|mythos)([/_.-]|$)/.test(id)
	);
}

function isTextOutputModel(model: RouteraModelRecord): boolean {
	const modality = model.architecture?.modality?.toLowerCase() ?? "";
	if (!modality) return true;

	// A modality such as text+image->image is an image-generation model, not a
	// text model pi can use for an agent turn. Keep vision models such as
	// text+image->text.
	const output = modality.split(/->|=>/).at(-1) ?? modality;
	return output.includes("text") || !/image|audio|video/.test(output);
}

function supportsImageInput(model: RouteraModelRecord): boolean {
	return model.architecture?.modality?.toLowerCase().includes("image") ?? false;
}

const REASONING_MODEL_PATTERNS: readonly RegExp[] = [
	/(?:^|[/_.-])o[134](?:[/_.-]|$)/i, // OpenAI o-series reasoning models
	/(?:^|[/_.-])gpt-5(?:[/_.-]|$)/i, // OpenAI GPT-5
	/(?:^|[/_.-])deepseek-r(?:easoner)?(?:[/_.-]|$)/i, // DeepSeek R-series + Reasoner
	/(?:^|[/_.-])kimi-k\d/i, // Kimi K-series reasoning models
	/(?:^|[/_.-])qwen[^/]*think/i, // Qwen thinking variants
];

function isReasoningModel(id: string): boolean {
	return REASONING_MODEL_PATTERNS.some((pattern) => pattern.test(id));
}

function publishedMaxTokens(record: RouteraModelRecord): number | undefined {
	const publishedMax = isPositiveNumber(record.max_output_tokens) ? record.max_output_tokens : record.max_tokens;
	return isPositiveNumber(publishedMax) ? Math.floor(publishedMax) : undefined;
}

function anthropicMaxTokens(record: RouteraModelRecord, contextWindow: number): number {
	return Math.min(
		contextWindow,
		publishedMaxTokens(record) ??
			(contextWindow >= 1_000_000 ? ANTHROPIC_1M_CONTEXT_MAX_TOKENS : ANTHROPIC_DEFAULT_MAX_TOKENS),
	);
}

function openAIMaxTokens(record: RouteraModelRecord, contextWindow: number, reasoning: boolean): number {
	return Math.min(
		contextWindow,
		publishedMaxTokens(record) ?? (reasoning ? OPENAI_REASONING_MAX_TOKENS : DEFAULT_MAX_TOKENS),
	);
}

// "off" is intentionally omitted from these maps: pi treats an absent `off`
// entry as a supported level (undefined passes the `!== null` selectable
// check), so the user can disable thinking. For OpenAI, off maps to no
// `reasoning_effort`; for Anthropic, off maps to `thinking: { type: "disabled" }`.
const THINKING_LEVEL_MAP_BASE = {
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
} as const;

const OPENAI_THINKING_LEVEL_MAP = { ...THINKING_LEVEL_MAP_BASE, xhigh: null, max: null } as const;
const ANTHROPIC_THINKING_LEVEL_MAP = { ...THINKING_LEVEL_MAP_BASE, xhigh: "xhigh", max: "max" } as const;

function mapModel(record: RouteraModelRecord, api: RouteraApi): Model<RouteraApi> {
	const contextWindow = isPositiveNumber(record.context_length)
		? Math.floor(record.context_length)
		: DEFAULT_CONTEXT_WINDOW;

	if (api === "anthropic-messages") {
		return {
			id: record.id,
			name: record.name ?? record.id,
			api,
			provider: PROVIDER_ID,
			baseUrl: ROUTERA_BASE_URL,
			reasoning: true,
			input: supportsImageInput(record) ? ["text", "image"] : ["text"],
			// Routera bills platform tokens rather than publishing USD rates for pi.
			// Keep costs honest instead of interpreting undocumented pricing units.
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow,
			maxTokens: anthropicMaxTokens(record, contextWindow),
			thinkingLevelMap: ANTHROPIC_THINKING_LEVEL_MAP,
			compat: {
				// Not every Routera Claude model supports adaptive thinking
				// (e.g. claude-haiku-4.5 rejects it with a 400), so leave
				// forceAdaptiveThinking unset and let pi use budget-based
				// thinking, which Routera's Anthropic endpoint accepts on every
				// model. Routera does not document the newer per-tool eager
				// streaming field, so disable it.
				supportsEagerToolInputStreaming: false,
			},
		};
	}

	const reasoning = isReasoningModel(record.id);
	return {
		id: record.id,
		name: record.name ?? record.id,
		api,
		provider: PROVIDER_ID,
		baseUrl: OPENAI_BASE_URL,
		reasoning,
		input: supportsImageInput(record) ? ["text", "image"] : ["text"],
		// Routera bills platform tokens rather than publishing USD rates for pi.
		// Keep costs honest instead of interpreting undocumented pricing units.
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: openAIMaxTokens(record, contextWindow, reasoning),
		...(reasoning ? { thinkingLevelMap: OPENAI_THINKING_LEVEL_MAP } : {}),
		compat: {
			// Routera documents system/messages and reasoning_effort, but not
			// OpenAI's developer role or store field.
			supportsDeveloperRole: false,
			supportsStore: false,
			maxTokensField: "max_completion_tokens",
			supportsReasoningEffort: reasoning,
		},
	};
}

async function fetchCatalog(
	label: string,
	url: string,
	apiKey: string,
	signal: AbortSignal,
): Promise<RouteraModelRecord[]> {
	const response = await fetch(url, {
		headers: {
			Accept: "application/json",
			Authorization: `Bearer ${apiKey}`,
		},
		signal: AbortSignal.any([signal, AbortSignal.timeout(DISCOVERY_TIMEOUT_MS)]),
	});
	if (!response.ok) {
		throw new Error(`Routera ${label} returned HTTP ${response.status} ${response.statusText}`);
	}

	const body = (await response.json()) as RouteraModelsResponse;
	return Array.isArray(body.data) ? body.data : [];
}

async function fetchRouteraModels(context: RefreshModelsContext): Promise<readonly Model<RouteraApi>[]> {
	const credential = context.credential;
	const apiKey = credential?.type === "api_key" ? credential.key : undefined;
	if (!apiKey) throw new Error("Routera API key is required to discover models");

	const [openaiRecords, anthropicRecords] = await Promise.all([
		fetchCatalog("models", OPENAI_MODELS_URL, apiKey, context.signal),
		fetchCatalog("Anthropic models", ANTHROPIC_MODELS_URL, apiKey, context.signal),
	]);

	const models = new Map<string, Model<RouteraApi>>();
	for (const record of openaiRecords) {
		if (!record.id || !isTextOutputModel(record)) continue;
		const api: RouteraApi = isAnthropicModel(record) ? "anthropic-messages" : "openai-completions";
		models.set(record.id, mapModel(record, api));
	}
	for (const record of anthropicRecords) {
		if (!record.id || !isTextOutputModel(record) || !isAnthropicModel(record)) continue;
		models.set(record.id, mapModel(record, "anthropic-messages"));
	}

	return [...models.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export default function (pi: ExtensionAPI): void {
	pi.registerProvider(
		createProvider<RouteraApi>({
			id: PROVIDER_ID,
			name: "Routera",
			baseUrl: OPENAI_BASE_URL,
			auth: { apiKey: bearerHeaderAuth(envApiKeyAuth("Routera API key", ["ROUTERA_API_KEY"])) },
			models: [],
			fetchModels: fetchRouteraModels,
			api: {
				"anthropic-messages": anthropicMessagesApi(),
				"openai-completions": openAICompletionsApi(),
			},
		}),
	);
}
