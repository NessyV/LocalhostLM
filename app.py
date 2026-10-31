import asyncio
import ast
import ipaddress
import json
import math
import os
import re
import socket
import sys
import webbrowser
from datetime import datetime, timezone
from html import unescape
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request as UrlRequest, urlopen
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from openai import AsyncOpenAI

app = FastAPI(title="Localhost.LM")

# Enable CORS for local cross-origin security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "histories")
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

SYSTEM_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "sysPrompt")
SYSTEM_PROMPT_FILE = os.path.join(SYSTEM_PROMPT_DIR, "system_prompts.json")
os.makedirs(SYSTEM_PROMPT_DIR, exist_ok=True)
if not os.path.exists(SYSTEM_PROMPT_FILE):
    with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as prompt_file:
        json.dump([], prompt_file, indent=2, ensure_ascii=False)
    
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", "q4_0")
# Point to local Ollama instance (using OpenAI-compatible endpoint)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_RAW_URL = os.environ.get("OLLAMA_RAW_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")

client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "llama3.1:latest")
SYSTEM_MESSAGE = os.environ.get("SYSTEM_MESSAGE", "").strip()

MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "2"))#do not change below 2, this to allow it to generate message and title at the same time
LOCAL_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

_server_stats: dict = {
    "max_slots": MAX_CONCURRENT_REQUESTS,
    "active": 0,
    "queued": 0,
}

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "32768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "16384"))
CONTEXT_SAFETY_TOKENS = int(os.environ.get("CONTEXT_SAFETY_TOKENS", "1024"))
MAX_INPUT_TOKENS = max(
    1024,
    MAX_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - CONTEXT_SAFETY_TOKENS,
)

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
DDG_TIMEOUT_SECONDS = float(os.environ.get("DDG_TIMEOUT_SECONDS", "3.5"))
DDG_MAX_CONTEXT_CHARS = int(os.environ.get("DDG_MAX_CONTEXT_CHARS", "3500"))
AGENT_MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "6"))
AGENT_RESULT_LIMIT = int(os.environ.get("AGENT_RESULT_LIMIT", "18000"))

class ThoughtFilter:
    """Buffer streamed chunks and hide content inside reasoning tags."""
    def __init__(self) -> None:
        self.buffer = ""
        self.in_thought = False
        self.current_end_marker: str | None = None
        self.first_token_received = False
        self.markers = [
            ("<think>", "</think>"),
            ("<|channel>thought", "<channel|>"),
            ("<|channel>", "<channel|>"),
        ]

    def process(self, chunk_text: str) -> str:
        self.buffer += chunk_text
        output = ""
        while self.buffer:
            if not self.in_thought:
                earliest_start_idx = -1
                matched_start = None
                matched_end = None
                for start, end in self.markers:
                    idx = self.buffer.find(start)
                    if idx != -1 and (
                        earliest_start_idx == -1 or idx < earliest_start_idx
                    ):
                        earliest_start_idx = idx
                        matched_start = start
                        matched_end = end
                if earliest_start_idx != -1:
                    output += self.buffer[:earliest_start_idx]
                    self.buffer = self.buffer[
                        earliest_start_idx + len(matched_start) :
                    ]
                    self.in_thought = True
                    self.current_end_marker = matched_end
                else:
                    max_possible_partial = 0
                    for start, _ in self.markers:
                        for i in range(1, len(start)):
                            if self.buffer.endswith(start[:i]):
                                max_possible_partial = max(max_possible_partial, i)
                                break
                    if max_possible_partial > 0:
                        output += self.buffer[:-max_possible_partial]
                        self.buffer = self.buffer[-max_possible_partial:]
                        break
                    output += self.buffer
                    self.buffer = ""
            else:
                if not self.current_end_marker:
                    self.in_thought = False
                    continue
                idx = self.buffer.find(self.current_end_marker)
                if idx != -1:
                    self.buffer = self.buffer[idx + len(self.current_end_marker) :]
                    self.in_thought = False
                    self.current_end_marker = None
                else:
                    max_possible_partial = 0
                    for i in range(1, len(self.current_end_marker)):
                        if self.buffer.endswith(self.current_end_marker[:i]):
                            max_possible_partial = max(max_possible_partial, i)
                            break
                    self.buffer = (
                        self.buffer[-max_possible_partial:]
                        if max_possible_partial > 0
                        else ""
                    )
                    break
        if output and not self.first_token_received:
            cleaned_output = output.lstrip()
            if cleaned_output:
                self.first_token_received = True
                return cleaned_output
            return ""
        return output

    def flush(self) -> str:
        if not self.in_thought and self.buffer:
            result = self.buffer
            self.buffer = ""
            if result and not self.first_token_received:
                cleaned = result.lstrip()
                if cleaned:
                    self.first_token_received = True
                    return cleaned
                return ""
            return result
        return ""

class ToolStreamFilter:
    """Stream normal text while swallowing <tool_call>...</tool_call> tags."""
    START = "<tool_call>"
    END = "</tool_call>"

    def __init__(self):
        self.pending = ""
        self.in_tool = False
        self.saw_tool = False
        self.closed_tool = False
        self.tool_payload = ""

    @staticmethod
    def _partial_suffix(value: str, marker: str) -> int:
        low_value = value.lower()
        low_marker = marker.lower()
        max_len = min(len(value), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if low_value.endswith(low_marker[:size]):
                return size
        return 0

    def process(self, text: str) -> str:
        if not text or self.closed_tool:
            return ""
        self.pending += text
        output = ""
        while self.pending and not self.closed_tool:
            if not self.in_tool:
                low = self.pending.lower()
                idx = low.find(self.START)
                if idx != -1:
                    output += self.pending[:idx]
                    self.pending = self.pending[idx + len(self.START):]
                    self.in_tool = True
                    self.saw_tool = True
                    continue
                keep = self._partial_suffix(self.pending, self.START)
                if keep:
                    output += self.pending[:-keep]
                    self.pending = self.pending[-keep:]
                else:
                    output += self.pending
                    self.pending = ""
                break
            low = self.pending.lower()
            idx = low.find(self.END)
            if idx != -1:
                self.tool_payload += self.pending[:idx]
                self.pending = ""
                self.in_tool = False
                self.closed_tool = True
                break
            keep = self._partial_suffix(self.pending, self.END)
            if keep:
                self.tool_payload += self.pending[:-keep]
                self.pending = self.pending[-keep:]
            else:
                self.tool_payload += self.pending
                self.pending = ""
            break
        return output

    def flush_visible(self) -> str:
        if self.saw_tool:
            self.pending = ""
            return ""
        result = self.pending
        self.pending = ""
        return result

    def get_call(self):
        if not self.saw_tool:
            return None
        return _parse_tool_payload(self.tool_payload)

def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""

def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _content_text(message.get("content")).strip()
    return ""

def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.3))

def _estimate_content_tokens(content) -> int:
    if isinstance(content, str):
        return _estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if not isinstance(item, dict):
                total += 8
                continue
            item_type = item.get("type")
            if item_type == "text":
                total += _estimate_text_tokens(str(item.get("text") or ""))
            elif item_type == "image_url":
                total += 1200
            else:
                total += _estimate_text_tokens(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                )
        return total
    return _estimate_text_tokens(str(content or ""))

def _estimate_message_tokens(message: dict) -> int:
    if not isinstance(message, dict):
        return 8
    role = str(message.get("role") or "")
    return 12 + _estimate_text_tokens(role) + _estimate_content_tokens(
        message.get("content")
    )

def _estimate_messages_tokens(messages: list) -> int:
    return 32 + sum(_estimate_message_tokens(m) for m in messages)

def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _estimate_text_tokens(text) <= max_tokens:
        return text
    max_chars = max(1, int(max_tokens * 3.0))
    if len(text) <= max_chars:
        return text
    marker = "\n\n[Earlier content truncated to fit context window.]\n\n"
    marker_chars = len(marker)
    if max_chars <= marker_chars + 64:
        return text[-max_chars:]
    return marker + text[-(max_chars - marker_chars) :]

def _truncate_message(message: dict, max_tokens: int) -> dict:
    msg = dict(message)
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = _truncate_text(content, max(1, max_tokens - 16))
        return msg
    if isinstance(content, list):
        copied = []
        image_cost = 0
        text_positions = []
        for item in content:
            if not isinstance(item, dict):
                copied.append(item)
                continue
            new_item = dict(item)
            copied.append(new_item)
            if new_item.get("type") == "image_url":
                image_cost += 1200
            elif new_item.get("type") == "text":
                text_positions.append(len(copied) - 1)
        text_budget = max(64, max_tokens - image_cost - 32)
        if text_positions:
            per_text = max(32, text_budget // len(text_positions))
            for pos in text_positions:
                copied[pos]["text"] = _truncate_text(
                    str(copied[pos].get("text") or ""), per_text
                )
        msg["content"] = copied
        return msg
    return msg

def fit_context(messages: list, budget: int = MAX_INPUT_TOKENS) -> list:
    source = [dict(m) for m in messages if isinstance(m, dict)]
    if not source:
        return []
    if _estimate_messages_tokens(source) <= budget:
        return source
    leading_indices = []
    for i, msg in enumerate(source):
        if msg.get("role") == "system":
            leading_indices.append(i)
        else:
            break
    latest_user_idx = None
    for i in range(len(source) - 1, -1, -1):
        if source[i].get("role") == "user":
            latest_user_idx = i
            break
    fixed = set(leading_indices)
    if latest_user_idx is not None:
        fixed.add(latest_user_idx)
    fixed_messages = [source[i] for i in sorted(fixed)]
    fixed_cost = _estimate_messages_tokens(fixed_messages)
    if fixed_cost > budget:
        min_user_budget = 768 if latest_user_idx is not None else 0
        if latest_user_idx is not None:
            system_cost = _estimate_messages_tokens(
                [source[i] for i in leading_indices]
            )
            user_budget = max(128, budget - system_cost - 64)
            source[latest_user_idx] = _truncate_message(
                source[latest_user_idx], user_budget
            )
        fixed_messages = [source[i] for i in sorted(fixed)]
        if _estimate_messages_tokens(fixed_messages) > budget and leading_indices:
            available_for_system = max(
                256,
                budget
                - min_user_budget
                - (
                    _estimate_message_tokens(source[latest_user_idx])
                    if latest_user_idx is not None
                    else 0
                ),
            )
            per_system = max(128, available_for_system // len(leading_indices))
            for idx in leading_indices:
                source[idx] = _truncate_message(source[idx], per_system)
    selected = set(fixed)
    for i in range(len(source) - 1, -1, -1):
        if i in selected:
            continue
        candidate_indices = sorted(selected | {i})
        candidate = [source[j] for j in candidate_indices]
        if _estimate_messages_tokens(candidate) <= budget:
            selected.add(i)
    result = [source[i] for i in sorted(selected)]
    while len(result) > 1 and _estimate_messages_tokens(result) > budget:
        removed = False
        for i, msg in enumerate(result):
            if msg.get("role") == "system" and i == 0:
                continue
            if latest_user_idx is not None and msg.get("role") == "user":
                if _content_text(msg.get("content")) == _latest_user_text(result):
                    continue
            result.pop(i)
            removed = True
            break
        if not removed:
            break
    return result

#should work finally, maybe ? I'll check it later, but for now, this is a good start
def _should_try_ddg(text: str) -> bool:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text or len(text) > 1200:
        return False
    lower = text.lower()
    non_search_starts = (
        "write ", "rewrite ", "draft ", "compose ", "translate ", "summarize ",
        "proofread ", "roleplay ", "role-play ", "create a story", "make a story",
        "debug ", "fix this code", "refactor ", "generate code", "write code",
    )
    if lower.startswith(non_search_starts):
        return False
    freshness_terms = (
        "latest", "current", "currently", "today", "tonight", "right now",
        "recent", "recently", "news", "price", "score", "status", "weather",
        "forecast", "flight status", "who is the", "what is the current",
        "as of", "this week", "this month", "this year",
    )
    #this should trigger the model to initiate the search, a little bit finnicky but I think it is good enough ? I might need to make a content aware trigger instead of just keyword though.
    if any(term in lower for term in freshness_terms):
        return True
    factual_question = re.match(
        r"^(who|what|when|where|which|how old|how many|how much|define|meaning of|capital of|population of)\b",
        lower,
    )
    return bool(factual_question and ("?" in text or len(text) <= 220))

def _flatten_related_topics(items, limit: int = 4) -> list[dict]:
    out: list[dict] = []
    def walk(values):
        for item in values or []:
            if len(out) >= limit:
                return
            if not isinstance(item, dict):
                continue
            nested = item.get("Topics")
            if isinstance(nested, list):
                walk(nested)
                continue
            text = str(item.get("Text") or "").strip()
            url = str(item.get("FirstURL") or "").strip()
            if text:
                out.append({"text": text, "url": url})
    walk(items)
    return out


def _ddg_request_sync(query: str) -> dict | None:
    params = urlencode({
        "q": query, "format": "json", "no_html": "1", "no_redirect": "1",
        "skip_disambig": "1", "t": "ollama_chat",
    })
    request = UrlRequest(
        f"https://api.duckduckgo.com/?{params}",
        headers={"Accept": "application/json", "User-Agent": "OLLAMA-FastAPI-Space/1.0"},
    )
    with urlopen(request, timeout=DDG_TIMEOUT_SECONDS) as response:
        raw = response.read(1_000_000)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    return data if isinstance(data, dict) else None

async def duckduckgo_context(query: str) -> str | None:
    if not _should_try_ddg(query):
        return None
    try:
        data = await asyncio.to_thread(_ddg_request_sync, query[:500])
    except Exception:
        return None
    if not data:
        return None
    answer = str(data.get("Answer") or "").strip()
    abstract = str(data.get("AbstractText") or data.get("Abstract") or "").strip()
    definition = str(data.get("Definition") or "").strip()
    heading = str(data.get("Heading") or "").strip()
    source_name = str(
        data.get("AbstractSource") or data.get("DefinitionSource")
        or "DuckDuckGo Instant Answers"
    ).strip()
    source_url = str(data.get("AbstractURL") or data.get("DefinitionURL") or "").strip()
    related = _flatten_related_topics(data.get("RelatedTopics"), limit=4)
    substantive = answer or abstract or definition
    if not substantive and len(related) < 2:
        return None
    lines = [
        "DuckDuckGo Instant Answer context (automatically retrieved because query appears factual):",
        f"Query: {query[:500]}",
    ]
    if heading: lines.append(f"Topic: {heading}")
    if answer: lines.append(f"Instant answer: {answer}")
    if abstract: lines.append(f"Summary: {abstract}")
    if definition: lines.append(f"Definition: {definition}")
    if source_name: lines.append(f"Source: {source_name}")
    if source_url: lines.append(f"Source URL: {source_url}")
    if related:
        lines.append("Related Instant Answer items:")
        for item in related:
            suffix = f" — {item['url']}" if item.get("url") else ""
            lines.append(f"- {item['text']}{suffix}")
    lines.append("Use this context only when relevant.")
    return "\n".join(lines)[:DDG_MAX_CONTEXT_CHARS]

TOOL_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.I)
AGENT_PROMPT = """You have server-side tools. Use a tool whenever it materially improves the answer, especially for:
- current facts, dates, news, weather, prices, scores
- calculations, time lookups, Wikipedia search, or reading a URL

To call a tool, output ONLY this exact wrapper:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

Available tools:
calculator {"expression":"sqrt(144)+5"}
get_current_time {"time_zone":"America/New_York"}
web_search {"query":"...","limit":8}
search_wikipedia {"query":"...","limit":6}
read_webpage {"url":"https://...","max_chars":12000}"""

#prompt injection hehe. Should allow the model to use the tools in a more structured way, and also to avoid hallucinating the results of the tools. This is a bit of a hack, but it should work.. right ?. "Should" is the key word here, because I have no idea if this will work or not. I will have to test it later, but for now, this is a good start.
AGENT_PRIORITY_PROMPT = """AGENT MODE — MANDATORY WORKFLOW. Follow this for EVERY user turn:

1. The server has ALREADY prefetched the current clock and a live web search
   for this query. Both are attached below as "SERVER PREFETCH" system messages.
   Treat the prefetched clock as the source of truth for "today" — NEVER guess
   the date from your training data.

2. Prefer facts from the prefetched web search over anything you remember.
   If the prefetched results are insufficient, use the tools (web_search,
   search_wikipedia, read_webpage) to fetch more. Do NOT fall back to internal
   knowledge silently.

3. Cite source URLs where applicable.

4. If the web returned nothing useful, say so explicitly and mark the answer
   as "based on internal knowledge — not verified against the live web".

Output tool calls ONLY in the <tool_call>...</tool_call> wrapper.
Do not write prose before a tool call."""

def _calc(expr):
    expr = str(expr or "").strip()
    if not expr or len(expr) > 300:
        raise ValueError("Invalid expression length.")
    funcs = {
        "abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
        "log10": math.log10, "floor": math.floor, "ceil": math.ceil, "pow": pow,
    }
    names = {"pi": math.pi, "e": math.e, **funcs}
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call,
        ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Unsupported expression.")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError("Unsupported name.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in funcs
        ):
            raise ValueError("Unsupported function.")
    return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, names))

def _time_tool(args):
    name = str(args.get("time_zone") or "").strip()
    try:
        tz = ZoneInfo(name) if name else timezone.utc
    except Exception:
        raise ValueError("Invalid IANA timezone.")
    d = datetime.now(tz)
    return json.dumps({
        "iso": d.isoformat(),
        "time_zone": name or "UTC",
        "formatted": d.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
    })

async def _wiki(query, limit=6):
    query = str(query or "").strip()
    if not query:
        raise ValueError("Wikipedia query required.")
    limit = max(1, min(int(limit or 6), 10))
    def run():
        params = urlencode({
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrlimit": limit, "prop": "extracts|info", "exintro": "1",
            "explaintext": "1", "exsentences": "5", "inprop": "url",
            "format": "json", "utf8": "1", "origin": "*",
        })
        req = UrlRequest(
            "https://en.wikipedia.org/w/api.php?" + params,
            headers={"User-Agent": "OLLAMA-Agent/1.0"},
        )
        with urlopen(req, timeout=8) as response:
            data = json.loads(response.read(2_000_000).decode("utf-8", "replace"))
        pages = list(data.get("query", {}).get("pages", {}).values())
        pages.sort(key=lambda p: p.get("index", 999999)) #why was this 999999 again ?
        out = []
        for item in pages[:limit]:
            title = str(item.get("title") or "")
            out.append({
                "title": title,
                "extract": str(item.get("extract") or "").strip(),
                "url": str(item.get("fullurl") or ("https://en.wikipedia.org/wiki/" + title.replace(" ", "_"))),
            })
        return json.dumps({"query": query, "results": out}, ensure_ascii=False)
    try:
        return await asyncio.to_thread(run)
    except Exception:
        return ""
#so that it wont accidentally read some spicy or weird sites, I've learned it the hard way. Also it wasn't supposed to read localhost
def _safe_url(raw):
    url = str(raw or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Public HTTP/HTTPS URL required.")
    if parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith(".local"):
        raise ValueError("Private/local URLs blocked.")
    try:
        for info in socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        ):
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise ValueError("Private/local URLs blocked.")
    except socket.gaierror:
        raise ValueError("Could not resolve host.")
    return url

async def _read_url(raw, max_chars=12000):
    url = _safe_url(raw)
    max_chars = max(500, min(int(max_chars or 12000), 30000))
    def run():
        req = UrlRequest(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 Ollama-Agent/1.0",
                "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5",
            },
        )
        with urlopen(req, timeout=8) as response:
            final = _safe_url(response.geturl())
            ctype = response.headers.get("content-type", "")
            raw_text = response.read(max_chars * 4).decode("utf-8", "replace")
        if "html" in ctype.lower():
            raw_text = re.sub(
                r"(?is)<(script|style|noscript|svg|canvas|iframe|nav|footer|form).*?>.*?</\1>",
                " ", raw_text,
            )
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
            title = (
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_match.group(1))).strip()
                if title_match else ""
            )
            text = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", raw_text)).strip()
            return json.dumps({"url": final, "title": title, "content": text[:max_chars]}, ensure_ascii=False)
        return json.dumps({"url": final, "content_type": ctype, "content": raw_text[:max_chars]}, ensure_ascii=False)
    return await asyncio.to_thread(run)

async def _web(query, limit=6):
    query = str(query or "").strip()
    if not query:
        raise ValueError("Search query required.")
    try:
        data = await asyncio.to_thread(_ddg_request_sync, query[:500])
    except Exception as exc:
        return "DuckDuckGo request failed: " + str(exc)
    if not data:
        return "No DuckDuckGo Instant Answer results."
    out = []
    if data.get("Answer"):
        out.append({"title": "Answer", "text": data["Answer"], "url": data.get("AbstractURL", "")})
    if data.get("AbstractText"):
        out.append({"title": data.get("Heading") or "Overview", "text": data["AbstractText"], "url": data.get("AbstractURL", "")})
    if data.get("Definition"):
        out.append({"title": "Definition", "text": data["Definition"], "url": data.get("DefinitionURL", "")})
    out.extend(_flatten_related_topics(data.get("RelatedTopics"), limit=max(1, min(int(limit or 6), 10))))
    limit = max(1, min(int(limit or 6), 10))
    return json.dumps({"query": query, "results": out[:limit]}, ensure_ascii=False)

async def _ddg_html_search(query: str, limit: int = 8) -> str:
    query = str(query or "").strip()
    if not query:
        return ""
    limit = max(1, min(int(limit or 8), 10))

    def clean_html(value: str) -> str:
        value = re.sub(r"(?is)<[^>]+>", " ", value or "")
        return re.sub(r"\s+", " ", unescape(value)).strip()

    def decode_ddg_url(href: str) -> str:
        href = unescape(href or "")
        if href.startswith("//"):
            href = "https:" + href
        try:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
        return href

    def run():
        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query[:500]})
        req = UrlRequest(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(req, timeout=8) as response:
            raw = response.read(2_000_000).decode("utf-8", "replace")
        blocks = re.findall(r'(?is)<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>', raw)
        results = []
        for block in blocks:
            link = re.search(
                r'(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
            )
            if not link:
                continue
            snippet = re.search(
                r'(?is)<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                block,
            )
            results.append({
                "title": clean_html(link.group(2)),
                "url": decode_ddg_url(link.group(1)),
                "snippet": clean_html(snippet.group(1)) if snippet else "",
            })
            if len(results) >= limit:
                break
        return json.dumps({"query": query, "results": results}, ensure_ascii=False) if results else ""
    return await asyncio.to_thread(run)

async def _web_search_bundle(query: str, limit: int = 8) -> str:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Search query required.")
    limit = max(1, min(int(limit or 8), 10))
    instant, ddg_html, wiki = await asyncio.gather(
        _web(query, limit), _ddg_html_search(query, limit),
        _wiki(query, min(limit, 6)), return_exceptions=True,
    )
    sections = []
    if isinstance(wiki, str) and wiki and '"results": []' not in wiki:
        sections.append("WIKIPEDIA LIVE RESULTS:\n" + wiki)
    if isinstance(ddg_html, str) and ddg_html:
        sections.append("DUCKDUCKGO WEB RESULTS:\n" + ddg_html)
    if (isinstance(instant, str) and instant
        and "No DuckDuckGo Instant Answer results" not in instant
        and "request failed" not in instant):
        sections.append("DUCKDUCKGO INSTANT ANSWERS:\n" + instant)
    stamp = datetime.now(timezone.utc).isoformat()
    if not sections:
        return json.dumps({"query": query, "retrieved_at": stamp, "results": [], "note": "All search sources returned no results."}, ensure_ascii=False)
    return (json.dumps({"query": query, "retrieved_at": stamp}, ensure_ascii=False)
            + "\n\n" + "\n\n".join(sections))[:20000]

async def _tool(name, args):
    args = args if isinstance(args, dict) else {}
    if name == "calculator":
        return _calc(args.get("expression", ""))
    if name == "get_current_time":
        return _time_tool(args)
    if name == "web_search":
        return await _web_search_bundle(args.get("query", ""), args.get("limit", 8))
    if name == "search_wikipedia":
        return await _wiki(args.get("query", ""), args.get("limit", 6))
    if name == "read_webpage":
        return await _read_url(args.get("url", ""), args.get("max_chars", 12000))
    raise ValueError("Unknown tool: " + str(name))

def _parse_tool_payload(payload):
    raw = str(payload or "").strip()
    if not raw:
        return None
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = str(fn.get("name") or "").strip()
        args = fn.get("arguments", {})
    else:
        name = str(obj.get("name") or obj.get("tool") or obj.get("tool_name") or "").strip()
        args = obj.get("arguments", obj.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            try:
                args = ast.literal_eval(args)
            except Exception:
                args = {}
    if not isinstance(args, dict):
        args = {}
    return (name, args) if name else None

def _needs_forced_web_lookup(text: str) -> bool:
    value = (text or "").lower().strip()
    if not value:
        return False
    freshness = (
        "today", "tonight", "tomorrow", "yesterday", "latest", "recent",
        "recently", "current", "currently", "right now", "news", "event",
        "weather", "forecast", "price", "stock",
    )
    return any(term in value for term in freshness)

def _build_contextual_search_query(messages: list):
    latest = _latest_user_text(messages)
    if not latest:
        return None
    if _needs_forced_web_lookup(latest):
        return latest[:500]
    return None

async def _forced_web_context(query: str) -> str:
    query = str(query or "").strip()
    stamp = datetime.now(timezone.utc).isoformat()
    if not query:
        return f"SERVER TOOL NOTICE ({stamp}): No usable query."
    try:
        results = await _web_search_bundle(query, 8)
    except Exception as exc:
        return f"SERVER TOOL NOTICE ({stamp}): Search failed: {exc}"
    return f"SERVER TOOL NOTICE ({stamp}): Executed search: {query}\n\n{results}"[:20000]

async def _agent_prefetch_context(latest_user_text: str) -> list[dict]:
    out: list[dict] = []
    try:
        now_utc = datetime.now(timezone.utc)
        clock_payload = {
            "iso": now_utc.isoformat(),
            "time_zone": "UTC",
            "formatted": now_utc.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
        }
        out.append({"role": "system", "content": (
            "SERVER PREFETCH — CURRENT CLOCK (authoritative; use this to "
            "interpret \"today\", \"now\", \"latest\", \"recently\", etc.):\n"
            + json.dumps(clock_payload, ensure_ascii=False)
        )})
    except Exception as exc:
        out.append({"role": "system", "content": f"SERVER PREFETCH — CLOCK FAILED: {exc}"})

    query = (latest_user_text or "").strip()
    if query:
        try:
            results = await _web_search_bundle(query[:500], 8)
            stamp = datetime.now(timezone.utc).isoformat()
            out.append({"role": "system", "content": (
                "SERVER PREFETCH — LIVE WEB SEARCH "
                "(prefer this over any internal knowledge):\n"
                f"Query: {query[:300]}\n"
                f"Retrieved at: {stamp}\n\n"
                f"{results}"
            )[:20000]})
        except Exception as exc:
            out.append({"role": "system", "content": (
                "SERVER PREFETCH — LIVE WEB SEARCH FAILED. "
                "You may still call web_search yourself if needed. "
                "Error: " + str(exc)
            )})
    return out

def _delta_event(text: str) -> str:
    payload = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(payload)}\n\n"

def _status_event(status: str) -> str:
    return f"data: {json.dumps({'status': status})}\n\n"

async def _agent_stream(messages, model_name: str, input_budget: int = MAX_INPUT_TOKENS):
    work = list(messages)
    insert_at = 0
    while insert_at < len(work) and work[insert_at].get("role") == "system":
        insert_at += 1
    work.insert(insert_at, {"role": "system", "content": AGENT_PROMPT})
    for _ in range(max(1, min(AGENT_MAX_ROUNDS, 10))):
        stream = await client.chat.completions.create(
            model=model_name, messages=fit_context(work, budget=input_budget),
            stream=True, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
        )
        thought_filter = ThoughtFilter()
        tool_filter = ToolStreamFilter()
        visible_prefix = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            piece = thought_filter.process(token)
            if not piece:
                continue
            safe = tool_filter.process(piece)
            if safe:
                visible_prefix += safe
                yield _delta_event(safe)
        leftover = thought_filter.flush()
        if leftover:
            safe = tool_filter.process(leftover)
            if safe:
                visible_prefix += safe
                yield _delta_event(safe)
        call = tool_filter.get_call()
        if not call:
            tail = tool_filter.flush_visible()
            if tail:
                yield _delta_event(tail)
            return
        name, args = call
        yield f"data: {json.dumps({'agent_status': f'Using {name}'})}\n\n"
        await asyncio.sleep(0)
        try:
            result = await _tool(name, args)
        except Exception as exc:
            result = "Tool error: " + str(exc)
        canonical = ("<tool_call>"
            + json.dumps({"name": name, "arguments": args}, ensure_ascii=False, separators=(",", ":"))
            + "</tool_call>")
        work += [
            {"role": "assistant", "content": (visible_prefix + "\n" if visible_prefix.strip() else "") + canonical},
            {"role": "system", "content": f"TOOL RESULT ({name}):\n{str(result)[:AGENT_RESULT_LIMIT]}"},
        ]
    work.append({"role": "system", "content": "Tool-round limit reached. Answer now using available results."})
    stream = await client.chat.completions.create(
        model=model_name, messages=fit_context(work, budget=input_budget),
        stream=True, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
    )
    thought_filter = ThoughtFilter()
    tool_filter = ToolStreamFilter()
    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        piece = thought_filter.process(token)
        if piece:
            safe = tool_filter.process(piece)
            if safe:
                yield _delta_event(safe)
    leftover = thought_filter.flush()
    if leftover:
        safe = tool_filter.process(leftover)
        if safe:
            yield _delta_event(safe)
    tail = tool_filter.flush_visible()
    if tail:
        yield _delta_event(tail)

# ---------------------------------------------------------------------------
# FastAPI Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_ui() -> str:
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    fallback_html = """<!DOCTYPE html>
    <html><head><title>Local</title></head>
    <body style="background:#05070b;color:#fff;font-family:sans-serif;padding:40px;text-align:center;">
        <h2>Local Backend Server Active</h2>
        <p>Place your <code>index.html</code> file in the same directory as <code>app.py</code> to view the interface.</p>
    </body></html>"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(fallback_html)
    return fallback_html

@app.get("/v1/models")
async def get_models():
    def fetch_tags():
        req = UrlRequest(f"{OLLAMA_RAW_URL}/api/tags")
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    try:
        raw_data = await asyncio.to_thread(fetch_tags)
        models = [{"id": m.get("name") or m.get("model"), "object": "model"}
                  for m in raw_data.get("models", [])]
        return JSONResponse({"object": "list", "data": models})
    except Exception:
        return JSONResponse(
            {"object": "list", "data": [{"id": DEFAULT_MODEL_ID, "object": "model"}]},
            status_code=200,
        )

@app.get("/v1/server-status")
async def get_server_status():
    max_slots = _server_stats["max_slots"]
    active = _server_stats["active"]
    queued = _server_stats["queued"]
    return JSONResponse({
        "max_slots": max_slots, "active": active, "queued": queued,
        "available": max(0, max_slots - active),
    })

# ---------------------------------------------------------------------------
# Server process memory (RSS) — used by the RAM chip in the UI
# ---------------------------------------------------------------------------
@app.get("/v1/server-memory")
async def get_server_memory():
    """Return system RAM usage and the server process's RAM usage."""
    rss_mb: float | None = None
    vms_mb: float | None = None
    percent: float | None = None
    total_mb: float | None = None
    used_mb: float | None = None
    available_mb: float | None = None
    system_percent: float | None = None

    # 1. psutil (best — gives full system + process stats)
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        vms_mb = mem.vms / (1024 * 1024)
        try:
            percent = float(proc.memory_percent())
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            total_mb = vm.total / (1024 * 1024)
            used_mb = vm.used / (1024 * 1024)
            available_mb = vm.available / (1024 * 1024)
            system_percent = float(vm.percent)
        except Exception:
            pass
    except Exception:
        # 2. /proc/self/status + /proc/meminfo (Linux fallback)
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024
                    elif line.startswith("VmSize:"):
                        vms_mb = int(line.split()[1]) / 1024

            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = {}
                    for line in f:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meminfo[k.strip()] = v.strip()
                    total_kb = int(meminfo["MemTotal"].split()[0]) if "MemTotal" in meminfo else 0
                    avail_kb = int(meminfo["MemAvailable"].split()[0]) if "MemAvailable" in meminfo else 0
                    if total_kb > 0:
                        total_mb = total_kb / 1024
                        if avail_kb > 0:
                            available_mb = avail_kb / 1024
                            used_mb = total_mb - available_mb
                            system_percent = (used_mb / total_mb) * 100
            except Exception:
                pass

            if rss_mb is not None and total_mb:
                percent = (rss_mb / total_mb) * 100
        except Exception:
            # 3. stdlib resource module (last resort — process RSS only)
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == "darwin":
                    rss_mb = usage.ru_maxrss / (1024 * 1024)
                else:
                    rss_mb = usage.ru_maxrss / 1024
            except Exception:
                pass

    if os.name == "nt" and (total_mb is None or used_mb is None):
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MemoryStatusEx()
            memory_status.dwLength = ctypes.sizeof(MemoryStatusEx)
            global_memory_status = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            global_memory_status.argtypes = [ctypes.POINTER(MemoryStatusEx)]
            global_memory_status.restype = ctypes.c_int
            if global_memory_status(ctypes.byref(memory_status)):
                total_bytes = memory_status.ullTotalPhys
                available_bytes = memory_status.ullAvailPhys
                total_mb = total_bytes / (1024 * 1024)
                available_mb = available_bytes / (1024 * 1024)
                used_mb = total_mb - available_mb
                system_percent = float(memory_status.dwMemoryLoad)
        except Exception:
            pass

    return JSONResponse({
        "rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
        "vms_mb": round(vms_mb, 1) if vms_mb is not None else None,
        "total_mb": round(total_mb, 0) if total_mb is not None else None,
        "used_mb": round(used_mb, 0) if used_mb is not None else None,
        "available_mb": round(available_mb, 0) if available_mb is not None else None,
        "percent": round(percent, 2) if percent is not None else None,
        "system_percent": round(system_percent, 2) if system_percent is not None else None,
        "pid": os.getpid(),
    })
# ---------------------------------------------------------------------------
# Model metadata (context window + capabilities)
# ---------------------------------------------------------------------------
_MODEL_INFO_CACHE: dict[str, dict] = {}

def _fetch_model_show_sync(model_name: str) -> dict:
    payload = json.dumps({"name": model_name}).encode("utf-8")
    req = UrlRequest(
        f"{OLLAMA_RAW_URL}/api/show", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _extract_context_length(show_data: dict) -> int | None:
    model_info = show_data.get("model_info") or {}
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if str(key).endswith(".context_length"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    params = str(show_data.get("parameters") or "")
    for line in params.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("num_ctx"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    return None

async def get_model_show_info(model_name: str) -> dict:
    if not model_name:
        return {}
    cached = _MODEL_INFO_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        data = await asyncio.to_thread(_fetch_model_show_sync, model_name)
    except Exception:
        data = {}
    info = {
        "name": model_name,
        "capabilities": list(data.get("capabilities") or []),
        "context_length": _extract_context_length(data),
    }
    _MODEL_INFO_CACHE[model_name] = info
    return info

def _compute_input_budget(context_length: int | None) -> int:
    ctx = context_length if (context_length and context_length > 0) else MAX_CONTEXT_TOKENS
    return max(1024, ctx - MAX_OUTPUT_TOKENS - CONTEXT_SAFETY_TOKENS)

@app.get("/v1/model-info")
async def get_model_info(name: str = Query("")):
    info = await get_model_show_info(name)
    return JSONResponse(info or {"name": name, "capabilities": [], "context_length": None})

def _history_file_path(chat_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(chat_id or "default"))
    if not safe_id:
        safe_id = "default"
    return os.path.join(HISTORY_DIR, f"history_{safe_id}.json")

def _genericize_prompt_text(text: str) -> str:
    return re.sub(r"\bXortron\b", "Localhost.LM", text, flags=re.IGNORECASE)

def _normalize_system_prompts(prompts: list) -> list[dict]:
    normalized = []
    for item in prompts:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = _genericize_prompt_text(item["text"].strip())
        if not text:
            continue
        prompt = {
            "id": str(item.get("id") or f"sys_{len(normalized) + 1}"),
            "text": text,
            "enabled": item.get("enabled") is True,
        }
        if item.get("recoveredFromHistory") is True:
            prompt["recoveredFromHistory"] = True
        normalized.append(prompt)
    return normalized

@app.get("/v1/system-prompts")
async def get_system_prompts():
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as prompt_file:
            data = json.load(prompt_file)
    except (OSError, json.JSONDecodeError):
        data = []
    return JSONResponse(_normalize_system_prompts(data) if isinstance(data, list) else [])

@app.post("/v1/system-prompts")
async def save_system_prompts(request: Request):
    data = await request.json()
    if not isinstance(data, list):
        return JSONResponse({"detail": "Expected a list of system prompts."}, status_code=400)

    prompts = _normalize_system_prompts(data)
    temporary_file = SYSTEM_PROMPT_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as prompt_file:
        json.dump(prompts, prompt_file, indent=2, ensure_ascii=False)
    os.replace(temporary_file, SYSTEM_PROMPT_FILE)
    return JSONResponse({"status": "success", "count": len(prompts)})

@app.get("/v1/history")
async def get_history(chatId: str = Query("default")):
    file_path = _history_file_path(chatId)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return JSONResponse(data)
        except Exception:
            pass
    return JSONResponse({"SysPrompt": [], "prompts": [], "responses": [], "activePath": []})

@app.post("/v1/history")
async def save_history(request: Request, chatId: str = Query("default")):
    data = await request.json()
    file_path = _history_file_path(chatId)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return JSONResponse({"status": "success"})

@app.delete("/v1/history/all")
async def delete_all_history():
    removed = 0
    try:
        if os.path.isdir(HISTORY_DIR):
            for name in os.listdir(HISTORY_DIR):
                if name.startswith("history_") and name.endswith(".json"):
                    try:
                        os.remove(os.path.join(HISTORY_DIR, name))
                        removed += 1
                    except OSError:
                        continue
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)
    return JSONResponse({"status": "success", "removed": removed})

@app.delete("/v1/history")
async def delete_history(chatId: str = Query("default")):
    file_path = _history_file_path(chatId)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return JSONResponse({"status": "success", "deleted": True})
        except OSError as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)
    return JSONResponse({"status": "success", "deleted": False})

@app.post("/v1/chat/completions")
async def chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    model_name = body.get("model") or DEFAULT_MODEL_ID
    raw_messages = body.get("messages", [])
    base_messages = list(raw_messages) if isinstance(raw_messages, list) else []
    agent_enabled = body.get("agent", False) is True

    if SYSTEM_MESSAGE and not any(
        isinstance(message, dict) and message.get("role") == "system"
        for message in base_messages
    ):
        base_messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})

    async def send_model_stream(messages, input_budget: int):
        model_messages = fit_context(messages, budget=input_budget)
        stream = await client.chat.completions.create(
            model=model_name, messages=model_messages, stream=True,
            temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
        )
        thought_filter = ThoughtFilter()
        tool_filter = ToolStreamFilter()
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            filtered = thought_filter.process(token)
            if not filtered:
                continue
            safe = tool_filter.process(filtered)
            if safe:
                yield _delta_event(safe)
        leftover = thought_filter.flush()
        if leftover:
            safe = tool_filter.process(leftover)
            if safe:
                yield _delta_event(safe)
        tail = tool_filter.flush_visible()
        if tail:
            yield _delta_event(tail)

    async def event_generator():
        _server_stats["queued"] += 1
        entered = False
        try:
            if _server_stats["active"] >= _server_stats["max_slots"]:
                yield _status_event("queued")

            async with LOCAL_SEMAPHORE:
                _server_stats["queued"] -= 1
                entered = True
                _server_stats["active"] += 1
                try:
                    yield _status_event("streaming")

                    model_info = await get_model_show_info(model_name)
                    input_budget = _compute_input_budget(model_info.get("context_length"))
                    try:
                        messages = list(base_messages)
                        latest_text = _latest_user_text(messages)

                        if agent_enabled:
                            yield f"data: {json.dumps({'agent_status': 'Prefetching live clock'})}\n\n"
                            await asyncio.sleep(0)
                            try:
                                prefetch_msgs = await _agent_prefetch_context(latest_text)
                            except Exception as exc:
                                prefetch_msgs = [{"role": "system", "content": f"SERVER PREFETCH FAILED: {exc}"}]

                            yield f"data: {json.dumps({'agent_status': 'Live data ready'})}\n\n"
                            await asyncio.sleep(0)

                            insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                            messages.insert(insert_at, {"role": "system", "content": AGENT_PRIORITY_PROMPT})
                            for offset, m in enumerate(prefetch_msgs):
                                messages.insert(insert_at + 1 + offset, m)

                            urls = re.findall(r"https?://[^\s<>()\]\[\"']+", latest_text or "")
                            if urls:
                                yield f"data: {json.dumps({'agent_status': 'Reading webpage'})}\n\n"
                                await asyncio.sleep(0)
                                after_prefetch = insert_at + 1 + len(prefetch_msgs)
                                try:
                                    page_result = await _read_url(urls[0], 16000)
                                    messages.insert(after_prefetch, {"role": "system", "content": f"Webpage Content:\n\n{page_result}"})
                                    yield f"data: {json.dumps({'agent_status': 'Webpage read'})}\n\n"
                                    await asyncio.sleep(0)
                                except Exception as exc:
                                    messages.insert(after_prefetch, {"role": "system", "content": f"read_webpage failed: {exc}"})
                            messages = fit_context(messages, budget=input_budget)

                            async for event in _agent_stream(messages, model_name, input_budget):
                                yield event
                            yield "data: [DONE]\n\n"
                            return

                        if body.get("web", True) is not False:
                            ddg_context = await duckduckgo_context(latest_text)
                            if ddg_context:
                                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                                messages.insert(insert_at, {"role": "system", "content": ddg_context})

                        messages = fit_context(messages, budget=input_budget)
                        async for event in send_model_stream(messages, input_budget):
                            yield event
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        payload = {"choices": [{"delta": {"content": f"⚠️ Local Model Error: {str(error)}"}}]}
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                finally:
                    _server_stats["active"] -= 1
        finally:
            if not entered:
                _server_stats["queued"] -= 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port

if __name__ == "__main__":
    import uvicorn
    import threading
    import time

    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}/"

    print(f"\n==================================================")
    print(f"  Local Server Automatic Launch")
    print(f"  URL: {url}")
    print(f"==================================================\n")

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
    
#bohoo you reached the end, no idea why I've done that but congrats I guess.