// Provider mapping test with mocked Routera model catalogs.
// Run with: npm test

let pass = 0;
let fail = 0;
function assert(condition, message) {
	if (condition) pass++;
	else {
		fail++;
		console.error("  FAIL:", message);
	}
}
function assertEq(actual, expected, message) {
	const ok = actual === expected;
	if (ok) pass++;
	else {
		fail++;
		console.error(`  FAIL: ${message}\n        expected: ${JSON.stringify(expected)}\n        actual:   ${JSON.stringify(actual)}`);
	}
}

const originalFetch = globalThis.fetch;
const calls = [];
const catalogs = {
	openai: {
		object: "list",
		data: [
			{
				id: "openai/gpt-5.5",
				name: "GPT 5.5",
				owned_by: "openai",
				context_length: 1050000,
				architecture: { modality: "text+image->text" },
			},
			{
				id: "gpt-image-1.5",
				name: "GPT Image",
				owned_by: "openai",
				context_length: 128000,
				architecture: { modality: "text->image" },
			},
			{
				id: "qwen/qwen3.5-27b",
				name: "Qwen 3.5 27B",
				owned_by: "qwen",
				context_length: 262144,
				architecture: { modality: "text->text" },
			},
			{
				id: "moonshot/kimi-k3",
				name: "Kimi K3",
				owned_by: "moonshot",
				context_length: 1000000,
				max_output_tokens: 32768,
				architecture: { modality: "text->text" },
			},
			{
				id: "anthropic/claude-opus-4.8",
				name: "Claude Opus 4.8",
				owned_by: "anthropic",
				context_length: 1000000,
				architecture: { modality: "text+image->text" },
			},
		],
	},
	anthropic: {
		object: "list",
		data: [
			{
				id: "anthropic/claude-opus-4.8",
				name: "Claude Opus 4.8",
				owned_by: "anthropic",
				context_length: 1000000,
				architecture: { modality: "text+image->text" },
			},
			{
				id: "anthropic/claude-haiku-4.5",
				name: "Claude Haiku 4.5",
				owned_by: "anthropic",
				context_length: 200000,
				architecture: { modality: "text+image->text" },
			},
			// This verifies that a non-Anthropic model in the compatibility
			// response is not accidentally sent to the Anthropic endpoint.
			{
				id: "moonshot/kimi-k3",
				name: "Kimi K3",
				owned_by: "moonshot",
				context_length: 1000000,
				architecture: { modality: "text->text" },
			},
		],
	},
};

globalThis.fetch = async (url, init) => {
	calls.push({ url: String(url), init });
	return {
		ok: true,
		status: 200,
		statusText: "OK",
		async json() {
			return String(url).includes("compat=anthropic") ? catalogs.anthropic : catalogs.openai;
		},
	};
};

try {
	const mod = await import("../index.ts");
	assert(typeof mod.default === "function", "default export is a function");
	assert(mod.isAnthropicModel({ id: "anthropic/claude-opus-4.8" }), "prefixed Claude is Anthropic");
	assert(mod.isAnthropicModel({ id: "claude-sonnet-4.6", owned_by: "anthropic" }), "owned_by marks Anthropic");
	assert(!mod.isAnthropicModel({ id: "openai/gpt-5.5", owned_by: "openai" }), "OpenAI is not Anthropic");

	const registered = [];
	const pi = {
		registerProvider(provider) {
			registered.push(provider);
		},
		on() {},
	};

	await mod.default(pi);

	assertEq(registered.length, 1, "exactly one provider registered");
	const provider = registered[0];
	assertEq(provider.id, "routera", "provider id");
	assertEq(provider.name, "Routera", "provider name");
	assertEq(provider.baseUrl, "https://api.routera.one/v1", "provider base URL");
	assert(typeof provider.auth?.apiKey?.login === "function", "provider supports /login");
	assert(typeof provider.auth?.apiKey?.resolve === "function", "provider resolves auth");
	assert(typeof provider.refreshModels === "function", "provider refreshes models dynamically");
	assert(typeof provider.streamSimple === "function", "provider exposes streamSimple");

	const published = [];
	await provider.refreshModels({
		credential: { type: "api_key", key: "rta_test_key" },
		allowNetwork: true,
		force: true,
		signal: new AbortController().signal,
		publish: async (publication) => {
			published.push(publication);
			publication.update?.();
			return true;
		},
	});

	assertEq(calls.length, 2, "both compatibility catalogs fetched");
	assert(calls.some((call) => call.url === "https://api.routera.one/v1/models"), "OpenAI catalog URL");
	assert(calls.some((call) => call.url === "https://api.routera.one/v1/models?compat=anthropic"), "Anthropic catalog URL");
	for (const call of calls) {
		assertEq(call.init.headers.Authorization, "Bearer rta_test_key", "catalog uses bearer auth");
	}
	assert(published.length === 1, "catalog published once");

	const models = provider.getModels();
	assertEq(models.length, 5, "text models discovered and image generation filtered");
	const gpt = models.find((model) => model.id === "openai/gpt-5.5");
	const claude = models.find((model) => model.id === "anthropic/claude-opus-4.8");
	const haiku = models.find((model) => model.id === "anthropic/claude-haiku-4.5");
	const qwen = models.find((model) => model.id === "qwen/qwen3.5-27b");
	const kimi = models.find((model) => model.id === "moonshot/kimi-k3");
	assert(!!gpt, "GPT model discovered");
	assert(!!claude, "Claude model discovered");
	assert(!!haiku, "Haiku model discovered");
	assert(!!qwen, "Qwen model discovered");
	assert(!!kimi, "Kimi model discovered");
	assertEq(gpt?.api, "openai-completions", "GPT uses OpenAI API");
	assertEq(gpt?.baseUrl, "https://api.routera.one/v1", "GPT uses OpenAI base URL");
	assertEq(gpt?.compat?.supportsDeveloperRole, false, "GPT uses system role");
	assertEq(claude?.api, "anthropic-messages", "Claude uses Anthropic API");
	assertEq(claude?.baseUrl, "https://api.routera.one", "Claude uses Anthropic base URL without /v1");
	assertEq(claude?.compat?.forceAdaptiveThinking, undefined, "Claude uses pi budget-based thinking (Routera adaptive not universal)");
	assert(claude?.compat?.supportsEagerToolInputStreaming === false, "Claude disables eager tool input streaming");
	assert(claude?.thinkingLevelMap?.off === undefined, "Claude 'off' is a selectable level (omitted, not null)");
	assertEq(gpt?.maxTokens, 128000, "OpenAI reasoning models get the 128k output budget");
	assertEq(qwen?.maxTokens, 16384, "non-reasoning OpenAI models keep the default output budget");
	assertEq(kimi?.maxTokens, 32768, "published max_output_tokens wins over the reasoning budget");
	assertEq(claude?.maxTokens, 128000, "1M Claude models use Anthropic's synchronous 128k output budget");
	assertEq(haiku?.maxTokens, 64000, "200k Claude models keep the 64k output budget");
	assertEq(claude?.contextWindow, 1000000, "context window mapped");
	assert(Array.isArray(claude?.input) && claude.input.includes("image"), "vision input mapped");
	assertEq(claude?.cost.input, 0, "undocumented Routera pricing units are not guessed");
	assertEq(kimi?.api, "openai-completions", "non-Anthropic compatibility entry routed to OpenAI API, not Anthropic");


	const auth = provider.auth.apiKey;
	const login = await auth.login({
		signal: new AbortController().signal,
		prompt: async () => "rta_from_login",
		notify() {},
	});
	assertEq(login.type, "api_key", "login returns api_key credential");
	assertEq(login.key, "rta_from_login", "login returns prompted key");
	const resolved = await auth.resolve({
		credential: { type: "api_key", key: "rta_stored" },
		ctx: { env: async () => undefined },
		signal: new AbortController().signal,
	});
	assertEq(resolved.auth.apiKey, "rta_stored", "stored key wins");
	assertEq(
		resolved.auth.headers?.Authorization,
		"Bearer rta_stored",
		"Bearer header declared for stored key (Routera's canonical auth header)",
	);
	const resolvedEnv = await auth.resolve({
		credential: undefined,
		ctx: { env: async (name) => (name === "ROUTERA_API_KEY" ? "rta_env" : undefined) },
		signal: new AbortController().signal,
	});
	assertEq(resolvedEnv.auth.apiKey, "rta_env", "env key used when no stored credential");
	assertEq(resolvedEnv.auth.headers?.Authorization, "Bearer rta_env", "Bearer header declared for env key");
} finally {
	globalThis.fetch = originalFetch;
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
