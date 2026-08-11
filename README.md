# pi-routera-provider

A [pi](https://github.com/earendil-works/pi) extension for
[Routera.one](https://www.routera.one). It discovers Routera's compatibility
catalogs and sends each model through the endpoint format it requires:

- **Anthropic models** use Routera's Anthropic Messages API at
  `https://api.routera.one/v1/messages`.
- **All other models** use Routera's OpenAI-compatible Chat Completions API at
  `https://api.routera.one/v1/chat/completions`.

## Install

### As a pi package

```sh
pi install git:github.com/perezdap/pi-routera-provider
```

### Global, manual

```powershell
git clone https://github.com/perezdap/pi-routera-provider "$env:USERPROFILE\.pi\agent\extensions\routera"
```

### Quick test

```sh
pi -e ./index.ts
```

## Use

```text
/login routera
/model routera/<model-id>
```

Create a key from [Routera Account > API Keys](https://www.routera.one/account?tab=apiKeys),
or set `ROUTERA_API_KEY` before starting pi. The key is required for model
catalog discovery as well as generation requests. If you log in after pi has
started and the catalog is still empty, run `/reload`.

Example model ids include:

```text
routera/anthropic/claude-opus-4.8
routera/openai/gpt-5.5
routera/moonshot/kimi-k3
```

The model id after `routera/` is the id returned by Routera; do not remove its
vendor prefix.

## How it works

- The OpenAI catalog is fetched from
  `GET https://api.routera.one/v1/models`.
- The Anthropic catalog is fetched from
  `GET https://api.routera.one/v1/models?compat=anthropic`.
- Models are registered in one mixed-API provider. Claude/vendor-Anthropic
  models use `anthropic-messages`; every other model uses
  `openai-completions`.
- Routera's Anthropic endpoint is configured with the base URL
  `https://api.routera.one` because the Anthropic client appends `/v1/messages`.
- Routera's canonical auth is `Authorization: Bearer <key>`; the provider
  declares that header during auth resolution so both API families send it.
  Anthropic requests additionally include the `x-api-key` header added by pi's
  Anthropic client, which Routera also accepts.
- Routera bills platform tokens rather than documenting USD model rates, so pi
  cost metadata is intentionally zero instead of guessing the units of the
  catalog's pricing fields.

Routera currently documents a narrow OpenAI-compatible request surface
(`model`, `messages`, `stream`, and `reasoning_effort`). Tool calling and other
fields are delegated to pi's standard API implementations and depend on what
Routera accepts for the selected upstream model.

## Development

The project has peer dependencies on the pi packages already provided by pi.
For a local test checkout, make those packages resolvable from `node_modules`
(or run it from an environment where pi exposes them), then run:

```sh
npm test
```

The test uses mocked model catalogs and does not make an authenticated Routera
request. Node >= 22.6 is required so the test runner can import `index.ts`
directly (and for `AbortSignal.any`).
