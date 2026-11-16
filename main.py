# free-model-aggregator-02: one server, two client wires, real free models.
#
# A quality-ranked aggregator for OpenRouter's free models, dialled through the
# grader's gateway. It serves the OpenAI Chat Completions API and the Anthropic
# Messages API, in JSON and in server-sent events, and answers every prompt with
# a REAL completion from a live free model -- never an answer of its own.
#
# Design (Python 3.13 standard library only, one file: `python3 -P -s` keeps
# /bot off sys.path, so no sibling imports):
#   - The brief at $MAKEFUN_PROMPT_FILE is the only source of the model list,
#     the ratings and the prompt manifest. The catalog is built from it, and
#     --list prints the same document the server registers, byte for byte.
#   - Routing is by rating: for each request the server derives the prompt's
#     requirements from the request body itself (tools declared, vision parts,
#     JSON-mode response_format) and dials the highest-rated free model that
#     meets them. Requirements from the body, not from arrival order, so the
#     same request always routes the same way on a replay.
#   - Failover: a 429 or 5xx is retried once on the same model, then the server
#     walks down the rating order. Every refusal is reported in the
#     x-makefun-upstream-failures header (max 8 rows), so the failover column
#     can see a refusal the server recovered from.
#   - Determinism: every choice derives from the brief, the request bodies and
#     the gateway's answers. No wall clock and no unseeded randomness reach a
#     request; model ids, timestamps in listings and event ids are fixed or
#     taken from the gateway's own answers, so a replay leaves the same export.
#   - The gateway is the only route a model is dialled on: POST
#     $OPENAI_BASE_URL/chat/completions. An Anthropic-shaped request is
#     translated into a Chat Completions body first, and the gateway's answer
#     translated back. The model field on every answer and every streamed chunk
#     is the id actually dialled, with the gateway's openrouter/ prefix kept.
#   - The server polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says done.

import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

# The wire protocol's version, in the register body.
PROTOCOL_VERSION = 1
# How often GET /work is polled, in seconds.
POLL_INTERVAL_SEC = 0.1
# The most bytes of one client request body this server reads.
MAX_BODY_BYTES = 8 * 1024 * 1024
# One gateway call may take at most this long; the env waits 30 s per answer.
GATEWAY_TIMEOUT_SEC = 25
# The output cap when a request names none (the gateway refuses an uncapped call).
DEFAULT_MAX_TOKENS = 512
# The gateway's per-call output cap.
MAX_OUTPUT_TOKENS = 8192
# The most gateway calls spent on one prompt (6 prompts x 6 = 36 < the 60 cap).
MAX_ATTEMPTS_PER_PROMPT = 6
# Safety net: exit 0 rather than hang past the trial's wall if /work never says done.
WALL_GUARD_SEC = 220

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
# The trial's dummy key for the gateway lane; the gateway swaps it for the
# entrant's own pushed OpenRouter key, which never enters the sandbox.
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

SCHEMA_PREFIX = "free-model-aggregator-02/"
# The provider's public endpoint, for the catalog only: never the gateway's
# trial-local address, which differs between a live pass and its replay.
OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
# A fixed epoch second for every listed model's `created`: nothing here reads the clock.
FIXED_OPENAI_CREATED = 1767225600
# A fixed instant for every listed model's `created_at`.
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

Json = dict[str, object]


def log(message: str) -> None:
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief


def object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def model_of(value: object) -> Json | None:
    """One model row, as this server reads it."""
    if not is_record(value):
        return None
    raw = string_of(value.get("id"))
    if raw == "":
        raw = string_of(value.get("model"))
    if raw == "":
        return None
    named = string_of(value.get("provider"))
    if named == "":
        named = string_of(value.get("wire"))
    model_id = (
        f"openrouter/{raw}"
        if named == "openrouter" and not raw.lower().startswith("openrouter/")
        else raw
    )
    display_name = string_of(value.get("display_name"))
    rating = number_or_null(value.get("rating_milli"))
    return {
        "id": model_id,
        "provider": named if named != "" else "openrouter",
        "display_name": display_name if display_name != "" else raw,
        "rating_milli": rating if rating is not None else 0,
        "context_length": number_or_null(value.get("context_length")),
        "tools": bool_or_null(value.get("tools")),
        "vision": bool_or_null(value.get("vision")),
        "json": bool_or_null(value.get("json")),
        "free": value.get("free") is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row, as this server reads it (kept for the manifest's order)."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get("id"))
    if prompt_id == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": {
            "tools": requires.get("tools") is True,
            "vision": requires.get("vision") is True,
            "json": requires.get("json") is True,
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: {"models": [...], "prompts": [...], "error": ...}."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = documents_in(text)
    models: list[Json] = []
    prompts: list[Json] = []
    for doc in docs:
        schema = string_of(doc.get("schema"))
        if SCHEMA_PREFIX not in schema:
            continue
        if isinstance(doc.get("models"), list) and not models:
            models = [m for m in (model_of(v) for v in doc["models"]) if m is not None]
        elif isinstance(doc.get("ratings"), list) and not models:
            models = [m for m in (model_of(v) for v in doc["ratings"]) if m is not None]
        if isinstance(doc.get("prompts"), list) and not prompts:
            prompts = [p for p in (prompt_of(v) for v in doc["prompts"]) if p is not None]
    if not models:
        return {"models": [], "prompts": prompts, "error": "the brief holds no models document"}
    return {"models": models, "prompts": prompts, "error": None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in brief order."""
    return [m for m in brief["models"] if m["free"]]


def build_catalog(brief: Json) -> Json:
    """The catalog: one provider, exactly the brief's free models, once each."""
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief["models"]:
        model_id = model["id"]
        if not model["free"] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({"id": model_id, "free": True, "context_length": model["context_length"]})
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


def catalog_json() -> str:
    """The catalog document, the same string --list prints and register sends."""
    return json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)


# ---------------------------------------------------------------- routing


def capabilities_of(model: Json) -> Json:
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet a prompt's requirements."""
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    return True


def ranked_candidates(requires: Json) -> list[Json]:
    """Free models meeting `requires`, best rating first, id as the tie-break."""
    eligible = [
        m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))
    ]
    eligible.sort(key=lambda m: (-number_or_zero(m["rating_milli"]), str(m["id"])))
    return eligible


def requirements_from_openai(body: Json) -> Json:
    """A Chat Completions request's requirements, read off the body itself."""
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    response_format = body.get("response_format")
    response_type = response_format.get("type") if is_record(response_format) else None
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if is_record(part) and part.get("type") == "image_url":
                vision = True
    return {
        "tools": tools,
        "vision": vision,
        "json": response_type in ("json_object", "json_schema"),
    }


def requirements_from_anthropic(body: Json) -> Json:
    """A Messages request's requirements, read off the body itself."""
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") == "image":
                vision = True
    return {"tools": tools, "vision": vision, "json": False}


def declared_tool_names(body: Json, wire: str) -> set[str]:
    """The tool names the client declared, for the wire's tool-name check."""
    names: set[str] = set()
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for raw in tools:
        if not is_record(raw):
            continue
        if wire == "anthropic":
            if isinstance(raw.get("name"), str):
                names.add(raw["name"])
        else:
            fn = raw.get("function")
            if is_record(fn) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
    return names


def clamp_max_tokens(value: object) -> int:
    """An output cap the gateway accepts: present, positive, at most 8192."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MAX_TOKENS
    n = int(value)
    if n <= 0:
        return DEFAULT_MAX_TOKENS
    return min(n, MAX_OUTPUT_TOKENS)


# ---------------------------------------------------------------- the gateway


class DialError(Exception):
    """A gateway call that never got an HTTP answer (timeout, unreachable)."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


class Upstream:
    """One gateway answer: its status and a body read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer."""
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def chat_completions_url() -> str:
    """The gateway's Chat Completions route."""
    if GATEWAY_URL == "":
        return ""
    return f"{GATEWAY_URL}/chat/completions" if GATEWAY_URL.endswith("/v1") else f"{GATEWAY_URL}/v1/chat/completions"


def error_code_of(status: int, text: str) -> str:
    """A refusal's code, from its error body when it names one."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed):
        error = parsed.get("error")
        if is_record(error):
            for key in ("code", "type"):
                if isinstance(error.get(key), str) and error[key] != "":
                    return error[key]
        if isinstance(parsed.get("code"), str) and parsed["code"] != "":
            return parsed["code"]
    return f"http_{status}"


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming `model`."""
    url = chat_completions_url()
    if url == "":
        raise DialError(502, "gateway_unset")
    data = json.dumps({**body, "model": model}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, as the provider sent it.
        return Upstream(err.code, err)
    except (TimeoutError, urllib.error.URLError, OSError) as err:
        text = str(err).lower()
        if isinstance(err, TimeoutError) or "timed out" in text or "timeout" in text:
            raise DialError(504, "timeout")
        raise DialError(502, "upstream_unreachable")
    return Upstream(response.status, response)


def try_once(model: str, body: Json) -> tuple[Upstream | None, Json | None]:
    """One gateway call: (upstream, None) or (None, failure row)."""
    try:
        upstream = dial(model, body)
    except DialError as err:
        return None, {"status": err.status, "code": err.code, "model": model}
    if upstream.ok:
        return upstream, None
    text = upstream.text()
    row = {"status": upstream.status, "code": error_code_of(upstream.status, text), "model": model}
    return None, row


def dial_candidates(candidates: list[Json], make_body) -> object:
    """Yield one (model id, upstream, failure row) per gateway call, in rating order.

    Each model is tried once, retried once on a 429 (a transient rate limit),
    then the next best model is tried. The caller stops the iteration when a
    call answers well; the generator then ends. At most
    MAX_ATTEMPTS_PER_PROMPT calls in all, so a trial stays under its 60 cap.
    """
    attempts = 0
    for model in candidates:
        if attempts >= MAX_ATTEMPTS_PER_PROMPT:
            return
        model_id = str(model["id"])
        upstream, row = try_once(model_id, make_body(model_id))
        attempts += 1
        yield model_id, upstream, row
        if upstream is not None:
            continue
        if row is not None and row["status"] == 429 and attempts < MAX_ATTEMPTS_PER_PROMPT:
            upstream, row = try_once(model_id, make_body(model_id))
            attempts += 1
            yield model_id, upstream, row


# ---------------------------------------------------------------- the client wires


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request."""
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [
        {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
        for m in free_models(BRIEF)
    ]


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": error_type, "code": code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        b["text"]
        for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part, or None."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object, or {} when they are not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as a Chat Completions request naming `model`."""
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_record(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})

    out: Json = {"model": model, "messages": messages}
    out["max_tokens"] = clamp_max_tokens(body.get("max_tokens"))
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                        "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_gateway_body(body: Json, model: str) -> Json:
    """A Chat Completions request as the gateway call naming `model`."""
    out: Json = {"model": model, "messages": body.get("messages", [])}
    for key in ("temperature", "top_p", "stop", "tools", "tool_choice", "response_format", "seed", "frequency_penalty", "presence_penalty", "logit_bias"):
        if key in body:
            out[key] = body[key]
    cap = body.get("max_tokens", body.get("max_completion_tokens"))
    out["max_tokens"] = clamp_max_tokens(cap)
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body: Json, model: str, declared: set[str]) -> Json:
    """A Chat Completions answer as an Anthropic Messages object.

    Tool calls naming a tool the client did not declare are dropped: the
    wire check allows no tool name the client did not declare.
    """
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content: list[Json] = []
    text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        name = fn["name"] if isinstance(fn.get("name"), str) else ""
        if declared and name not in declared:
            continue
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": name,
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": f"msg_{body['id'] if isinstance(body.get('id'), str) else 'aggregate'}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": number_or_zero(usage.get("prompt_tokens")),
            "output_tokens": number_or_zero(usage.get("completion_tokens")),
        },
    }


def openai_answer_with_model(body: Json, model: str, declared: set[str]) -> Json:
    """A Chat Completions answer with `model` the id dialled, undeclared tool calls dropped."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if choices and is_record(choices[0]):
        choice = dict(choices[0])
        message = choice.get("message") if is_record(choice.get("message")) else {}
        calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        if declared and calls:
            kept = []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                if isinstance(call["function"].get("name"), str) and call["function"]["name"] in declared:
                    kept.append(call)
            message = {**message, "tool_calls": kept}
            choice = {**choice, "message": message}
            body = {**body, "choices": [choice, *choices[1:]]}
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    if not usage:
        body = {**body, "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}}
    return {**body, "model": model}


def answer_has_content(body: Json, declared: set[str]) -> bool:
    """Whether a Chat Completions answer carries text or a declared tool call."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if not choices or not is_record(choices[0]):
        return False
    message = choices[0].get("message") if is_record(choices[0].get("message")) else {}
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        return True
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        name = call["function"].get("name")
        if isinstance(name, str) and (not declared or name in declared):
            return True
    return False


# ---------------------------------------------------------------- SSE


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the data line."""
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`."""
    lines: list[str] = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f"data: {json.dumps({**chunk, 'model': model}, separators=(',', ':'), ensure_ascii=False)}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on blank lines."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, `model` the dialled id."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model: str, declared: set[str]):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start, content_block_start/delta/stop per text block or tool call,
    then message_delta with the stop reason and usage, then message_stop.
    Tool calls naming a tool the client did not declare are dropped.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    tool_names: dict[object, str] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        tool_names.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}},
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "[DONE]":
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else "end_turn"))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                input_tokens = number_or_zero(chunk["usage"].get("prompt_tokens"))
                output_tokens = number_or_zero(chunk["usage"].get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            out.extend(start_events())
            if isinstance(delta.get("content"), str) and delta["content"] != "":
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event("content_block_start", {"type": "content_block_start", "index": text_index, "content_block": {"type": "text", "text": ""}})
                    )
                out.append(
                    sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": delta["content"]}},
                    )
                )
            calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call["index"] if is_number(call.get("index")) else len(tool_indexes)
                fn = call.get("function") if is_record(call.get("function")) else {}
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                if upstream_index in tool_names:
                    if tool_names[upstream_index] == "":
                        continue
                elif name != "":
                    if declared and name not in declared:
                        tool_names[upstream_index] = ""
                        continue
                    tool_names[upstream_index] = name
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    out.append(
                        sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{index}",
                                    "name": tool_names.get(upstream_index, ""),
                                    "input": {},
                                },
                            },
                        )
                    )
                if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                    out.append(
                        sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                        )
                    )
            if choice.get("finish_reason") is not None:
                pending_stop = stop_reason_of(choice.get("finish_reason"))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else "end_turn"):
            yield event


# ---------------------------------------------------------------- the HTTP server


def failures_header(failures: list[Json] | None) -> str | None:
    """The x-makefun-upstream-failures header: at most 8 refusal rows, or None."""
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(",", ":"), ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/2"
    sys_version = ""

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing ---------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        header = failures_header(failures)
        if header is not None:
            self.send_header("x-makefun-upstream-failures", header)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, failures: list[Json] | None = None) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message), failures)
        else:
            self.send_json(status, openai_error(message, error_type, None), failures)

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        # Python's http.server does not chunk a response without a length; the
        # close is what ends the stream for the client.
        self.send_header("connection", "close")
        header = failures_header(failures)
        if header is not None:
            self.send_header("x-makefun-upstream-failures", header)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, failures: list[Json]) -> None:
        """No free model answered: a 502 in the client's wire, refusals reported."""
        last = failures[-1] if failures else {"status": 502, "code": "upstream_unreachable", "model": None}
        detail = f"the gateway answered HTTP {last['status']} ({last['code']})"
        if wire == "anthropic":
            self.send_error_body(wire, 502, "api_error", detail, failures)
        else:
            self.send_error_body(wire, 502, "upstream_error", detail, failures)

    # -- the routes -------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model."""
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requires = requirements_from_openai(body)
        candidates = ranked_candidates(requires)
        if not candidates:
            self.send_error_body("openai", 503, "no_free_model", "no free model in the brief meets this request")
            return
        declared = declared_tool_names(body, "openai")
        stream = body.get("stream") is True
        failures: list[Json] = []
        for model_id, upstream, row in dial_candidates(candidates, lambda m: openai_gateway_body(body, m)):
            if row is not None:
                failures.append(row)
            if upstream is None:
                continue
            if stream:
                self.stream_headers(failures)
                try:
                    for frame in openai_stream_with_model(upstream.chunks(), model_id):
                        self.write_event(frame)
                finally:
                    upstream.close()
                return
            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                failures.append({"status": 200, "code": "invalid_json", "model": model_id})
                continue
            if not is_record(parsed) or not answer_has_content(parsed, declared):
                failures.append({"status": 200, "code": "empty_answer", "model": model_id})
                continue
            self.send_json(200, openai_answer_with_model(parsed, model_id, declared), failures)
            return
        self.refused("openai", failures)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and back."""
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requires = requirements_from_anthropic(body)
        candidates = ranked_candidates(requires)
        if not candidates:
            self.send_error_body("anthropic", 503, "overloaded_error", "no free model in the brief meets this request")
            return
        declared = declared_tool_names(body, "anthropic")
        stream = body.get("stream") is True
        failures: list[Json] = []
        for model_id, upstream, row in dial_candidates(candidates, lambda m: anthropic_to_openai(body, m)):
            if row is not None:
                failures.append(row)
            if upstream is None:
                continue
            if stream:
                self.stream_headers(failures)
                try:
                    for event in anthropic_events_from_openai(upstream.chunks(), model_id, declared):
                        self.write_event(event)
                finally:
                    upstream.close()
                return
            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                failures.append({"status": 200, "code": "invalid_json", "model": model_id})
                continue
            if not is_record(parsed) or not answer_has_content(parsed, declared):
                failures.append({"status": 200, "code": "empty_answer", "model": model_id})
                continue
            self.send_json(200, openai_to_anthropic(parsed, model_id, declared), failures)
            return
        self.refused("anthropic", failures)


# ---------------------------------------------------------------- register and poll


def register(port: int) -> bool:
    """POST the port and catalog to the env; the env answers {"ok": true, "requests": 6}."""
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG}
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    for attempt in range(10):
        try:
            request = urllib.request.Request(
                f"{ENV_URL}/register",
                data=data,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                text = response.read().decode("utf-8", "replace")
            log(f"registered on port {port}: {text[:200]}")
            return True
        except Exception as err:
            log(f"register attempt {attempt + 1} failed: {err}")
            time.sleep(0.5)
    return False


def work_done() -> bool:
    """Whether GET $MAKEFUN_ENV_URL/work says the env has driven every prompt."""
    try:
        request = urllib.request.Request(f"{ENV_URL}/work", method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return is_record(payload) and payload.get("done") is True
    except Exception as err:
        log(f"work poll failed: {err}")
        return False


def main() -> int:
    if "--list" in sys.argv[1:]:
        print(catalog_json())
        return 0
    if BRIEF.get("error"):
        log(f"brief: {BRIEF['error']}")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    log(f"listening on 0.0.0.0:{port} ({len(served_models())} free models)")
    if ENV_URL != "":
        register(port)
    if ENV_URL == "":
        # Outside a trial: serve until stopped (handy for a local check).
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    start = time.monotonic()
    while True:
        if work_done():
            break
        if time.monotonic() - start > WALL_GUARD_SEC:
            log("wall guard reached; exiting 0")
            break
        time.sleep(POLL_INTERVAL_SEC)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
