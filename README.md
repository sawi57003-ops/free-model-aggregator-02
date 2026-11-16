# free-model-aggregator-02

One server, two client wires, real free models. It serves the OpenAI Chat
Completions API and the Anthropic Messages API (JSON and SSE) and answers
every prompt with a real completion from a live free OpenRouter model, dialled
through the grader's gateway on the entrant's pushed key.

## Run

```sh
python3 main.py            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers, serves, exits on done
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13 standard library only; nothing is installed.

## How it routes

- The brief is the only source of the model list and the ratings. The catalog
  is built from it, and `--list` prints the same document the server registers.
- Each request's requirements are read off the request body itself (tools
  declared, vision parts, JSON-mode `response_format`), and the server dials
  the highest-rated free model meeting them: Inkling 2 Small (1475) for plain
  and tools prompts, Nemotron 3.5 Super VL (1303) for JSON-mode prompts.
- A 429 is retried once on the same model, then the server walks down the
  rating order. Every refusal is reported in the
  `x-makefun-upstream-failures` header (max 8 rows).
- Every completion is dialled on `POST $OPENAI_BASE_URL/chat/completions`;
  Anthropic-shaped requests are translated there and back. The `model` field on
  every answer and streamed chunk is the id actually dialled.

## Local test

`../free-model-aggregator-02/mock_test.py` runs a mock env + gateway on
loopback and drives the six exact trial-0 prompt bodies: 28 checks, all
passing (catalog byte-equality, both `/v1/models` wires, token echo and model
field per prompt, the 429 failover header, SSE shape, exit 0 on done).
