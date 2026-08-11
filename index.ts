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
const DISCOVERY_TIMEOUT_MS = 15_000;

type RouteraApi = "anthropic-messages" | "openai-completions";

interface RouteraModelRecord {
	id: string;
	name?: string;
	owned_by?: string;
	context_length?: number;
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

function isReasoningModel(id: string): boolean {
	return /(?:^|[/_.-])(?:o[134](?:[/_.-]|$)|gpt-5(?:[/_.-]|$)|deepseek-r(?:easoner)?(?:[/_.-]|$)|kimi-k\d|qwen[^/]*think)/i.test(
		id,
	);
}

// "off" is intentionally omitted: pi treats an absent `off` entry as a
// supported level (undefined passes the `!== null` selectable check), so the
// user can disable thinking. For OpenAI, off maps to no `reasoning_effort`;
// for Anthropic, off maps to `thinking: { type: "disabled" }`.
const OPENAI_THINKING_LEVEL_MAP = {
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: null,
	max: null,
} as const;

const ANTHROPIC_THINKING_LEVEL_MAP = {
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: "xhigh",
	max: "max",
} as const;

function mapModel(record: RouteraModelRecord, api: RouteraApi): Model<RouteraApi> {
	const anthropic = api === "anthropic-messages";
	const contextWindow = isPositiveNumber(record.context_length)
		? Math.floor(record.context_length)
		: DEFAULT_CONTEXT_WINDOW;
	const reasoning = anthropic || isReasoningModel(record.id);

	return {
		id: record.id,
		name: record.name ?? record.id,
		api,
		provider: PROVIDER_ID,
		baseUrl: anthropic ? ROUTERA_BASE_URL : OPENAI_BASE_URL,
		reasoning,
		input: supportsImageInput(record) ? ["text", "image"] : ["text"],
		// Routera bills platform tokens rather than publishing USD rates for pi.
		// Keep costs honest instead of interpreting undocumented pricing units.
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: Math.min(contextWindow, DEFAULT_MAX_TOKENS),
		...(reasoning
			? {
				thinkingLevelMap: anthropic ? ANTHROPIC_THINKING_LEVEL_MAP : OPENAI_THINKING_LEVEL_MAP,
			}
			: {}),
		compat: anthropic
			? {
				// Not every Routera Claude model supports adaptive thinking
				// (e.g. claude-haiku-4.5 rejects it with a 400), so leave
				// forceAdaptiveThinking unset and let pi use budget-based
				// thinking, which Routera's Anthropic endpoint accepts on every
				// model. Routera does not document the newer per-tool eager
				// streaming field, so disable it.
				supportsEagerToolInputStreaming: false,
			}
			: {
				// Routera documents system/messages and reasoning_effort, but not
				// OpenAI's developer role or store field.
				supportsDeveloperRole: false,
				supportsStore: false,
				maxTokensField: "max_completion_tokens",
				supportsReasoningEffort: reasoning,
			},
	} as Model<RouteraApi>;
}

async function fetchCatalog(
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
		throw new Error(`Routera ${url.endsWith("models") ? "models" : "Anthropic models"} returned HTTP ${response.status} ${response.statusText}`);
	}

	const body = (await response.json()) as RouteraModelsResponse;
	return Array.isArray(body.data) ? body.data : [];
}

async function fetchRouteraModels(context: RefreshModelsContext): Promise<readonly Model<RouteraApi>[]> {
	const credential = context.credential;
	const apiKey = credential?.type === "api_key" ? credential.key : undefined;
	if (!apiKey) throw new Error("Routera API key is required to discover models");

	const [openaiRecords, anthropicRecords] = await Promise.all([
		fetchCatalog(OPENAI_MODELS_URL, apiKey, context.signal),
		fetchCatalog(ANTHROPIC_MODELS_URL, apiKey, context.signal),
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
			auth: { apiKey: envApiKeyAuth("Routera API key", ["ROUTERA_API_KEY"]) },
			models: [],
			fetchModels: fetchRouteraModels,
			api: {
				"anthropic-messages": anthropicMessagesApi(),
				"openai-completions": openAICompletionsApi(),
			},
		}),
	);

	// pi-ai's Anthropic client normally sends x-api-key. Routera documents
	// Authorization: Bearer as its canonical auth header; send both so the
	// provider works with the SDK and with Routera's documented auth contract.
	pi.on("before_provider_headers", (event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_ID || ctx.model.api !== "anthropic-messages") return;
		const apiKeyHeader = event.headers["x-api-key"] ?? event.headers["X-API-Key"];
		if (typeof apiKeyHeader === "string") event.headers.Authorization = `Bearer ${apiKeyHeader}`;
	});
}
