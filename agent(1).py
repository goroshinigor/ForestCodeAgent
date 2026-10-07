#!/usr/bin/env python3
"""Terminal coding agent v2: Ollama (OpenAI-compatible) + tools + skills + MCP.

Run:   python agent.py            (new session)
       python agent.py --resume   (continue last session)
       python agent.py --auto     (no confirmations: only inside a sandbox/VM!)

Commands inside the REPL: /compact  /skills  /mcp  exit
Needs: pip install openai mcp
"""
import asyncio
import atexit
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI

MODEL = os.environ.get("AGENT_MODEL", "coder-local")
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:11434/v1")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", os.getcwd())).resolve()
HOME_CFG = Path.home() / ".agent"
PROJ_CFG = WORKDIR / ".agent"
SESSION_DIR = PROJ_CFG / "sessions"

MAX_OUTPUT = 8000      # chars per tool result
MAX_STEPS = 30         # tool-loop iterations per user message
TRIM_AT = 50000        # chars of history: start trimming old tool outputs
SUMMARIZE_AT = 70000   # chars of history: summarize old turns with the model
KEEP_RECENT = 8        # trailing messages kept verbatim when compacting

AUTO = "--auto" in sys.argv
RESUME = "--resume" in sys.argv

client = OpenAI(base_url=BASE_URL, api_key="ollama")
files_read: set = set()

SAFE_CMD = re.compile(
    r"^(ls|pwd|cat|head|tail|wc|rg|grep|find|git (status|diff|log|branch)|php -v|python3? --version)\b"
)
SHELL_META = re.compile(r"[;&|><`]|\$\(")


# =====================================================================
# helpers
# =====================================================================
def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-half:]


def allowed_roots() -> list:
    return [WORKDIR, HOME_CFG / "skills"]


def safe_path(p: str) -> Path:
    full = (WORKDIR / p).resolve()
    for root in allowed_roots():
        root = root.resolve()
        if full == root or root in full.parents:
            return full
    raise ValueError(f"path outside allowed directories: {p}")


def confirm(prompt: str) -> bool:
    if AUTO:
        return True
    return input(f"\n  ? {prompt} [y/N] ").strip().lower() in ("y", "yes", "д", "да")


# =====================================================================
# built-in tools
# =====================================================================
def read_file(path: str, offset: int = 0, limit: int = 400) -> str:
    p = safe_path(path)
    if not p.is_file():
        return f"Error: {path} is not a file"
    files_read.add(str(p))
    lines = p.read_text(errors="replace").splitlines()
    out = "\n".join(lines[offset: offset + limit])
    if offset + limit < len(lines):
        out += f"\n...[{len(lines) - offset - limit} more lines, use offset={offset + limit}]"
    return out


def write_file(path: str, content: str) -> str:
    p = safe_path(path)
    if p.exists() and str(p) not in files_read:
        return "Error: file exists; read_file it first, or use edit_file"
    if not confirm(f"write {path} ({len(content)} chars)?"):
        return "Denied by user"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    files_read.add(str(p))
    return f"Wrote {path}"


def edit_file(path: str, old: str, new: str) -> str:
    p = safe_path(path)
    if not p.is_file():
        return f"Error: {path} not found"
    if str(p) not in files_read:
        return "Error: read_file this file before editing it"
    text = p.read_text()
    count = text.count(old)
    if count == 0:
        return "Error: 'old' not found (must match exactly, including whitespace)"
    if count > 1:
        return f"Error: 'old' matches {count} places; include more context to make it unique"
    print(f"\n  --- edit {path}\n  - " + old.replace("\n", "\n  - ") + "\n  + " + new.replace("\n", "\n  + "))
    if not confirm("apply edit?"):
        return "Denied by user"
    p.write_text(text.replace(old, new, 1))
    return f"Edited {path}"


def list_files(pattern: str = "**/*") -> str:
    skip = {".git", "node_modules", "vendor", ".agent", "__pycache__", ".venv"}
    res = []
    for f in WORKDIR.glob(pattern):
        if f.is_file() and not (set(f.relative_to(WORKDIR).parts) & skip):
            res.append(str(f.relative_to(WORKDIR)))
        if len(res) >= 300:
            break
    return "\n".join(sorted(res)) or "(no matches)"


def grep(pattern: str, path: str = ".") -> str:
    target = str(safe_path(path))
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--max-count", "20", pattern, target]
    else:
        cmd = ["grep", "-rn", "--exclude-dir=.git", "--exclude-dir=node_modules", pattern, target]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKDIR)
    return truncate(r.stdout or r.stderr or "(no matches)")


def bash(command: str, timeout: int = 60) -> str:
    safe = SAFE_CMD.match(command.strip()) and not SHELL_META.search(command)
    if not safe and not confirm(f"run: {command}"):
        return "Denied by user"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           cwd=WORKDIR, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {timeout}s"
    return truncate(f"exit={r.returncode}\n{r.stdout}{r.stderr}")


def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
BASE_TOOLS = [
    _schema("read_file", "Read a text file (400 lines per call).", {"path": S, "offset": I, "limit": I}, ["path"]),
    _schema("write_file", "Create a new file or overwrite one you have read.", {"path": S, "content": S}, ["path", "content"]),
    _schema("edit_file", "Replace a UNIQUE exact string in a file you have read.", {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    _schema("list_files", "List files by glob pattern, e.g. 'src/**/*.php'.", {"pattern": S}, []),
    _schema("grep", "Search file contents by regex.", {"pattern": S, "path": S}, ["pattern"]),
    _schema("bash", "Run a shell command in the project dir.", {"command": S, "timeout": I}, ["command"]),
]
IMPL = {f.__name__: f for f in (read_file, write_file, edit_file, list_files, grep, bash)}


# =====================================================================
# skills: folders with SKILL.md (frontmatter: name, description)
# only name+description go into the prompt; the body is loaded on demand
# =====================================================================
def load_skills() -> dict:
    skills = {}
    for root in (HOME_CFG / "skills", PROJ_CFG / "skills"):   # project overrides global
        for f in sorted(root.glob("*/SKILL.md")):
            text = f.read_text(errors="replace")
            m = re.match(r"---\s*\n(.*?)\n---\s*\n(.*)", text, re.S)
            meta, body = (m.group(1), m.group(2)) if m else ("", text)
            n = re.search(r"^name:\s*(.+)$", meta, re.M)
            d = re.search(r"^description:\s*(.+)$", meta, re.M)
            name = n.group(1).strip().strip("\"'") if n else f.parent.name
            skills[name] = {
                "desc": d.group(1).strip().strip("\"'") if d else "(no description)",
                "body": body.strip(),
                "dir": str(f.parent),
            }
    return skills


SKILLS = load_skills()


def use_skill(name: str) -> str:
    s = SKILLS.get(name)
    if not s:
        return f"Error: unknown skill '{name}'. Available: {', '.join(SKILLS) or 'none'}"
    return f"# Skill: {name}\nResources directory: {s['dir']}\n\n{truncate(s['body'], 12000)}"


if SKILLS:
    BASE_TOOLS.append(_schema("use_skill", "Load full instructions of a skill by name.", {"name": S}, ["name"]))
    IMPL["use_skill"] = use_skill


# =====================================================================
# MCP client (stdio servers). Runs an asyncio loop in a background thread.
# Config: .agent/mcp.json (project) and ~/.agent/mcp.json (global)
#   {"servers": {"name": {"command": "npx", "args": [...], "env": {...},
#                          "trust": false, "tools": ["only_these"]}}}
# =====================================================================
class MCPManager:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.sessions, self.cfg, self.route = {}, {}, {}
        self.schemas, self.errors, self.stops = [], {}, []
        atexit.register(self.shutdown)

    async def _serve(self, name, cfg, ready):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        stop = asyncio.Event()
        self.stops.append(stop)
        try:
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})})
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    allow = cfg.get("tools")
                    for t in listed.tools:
                        if allow and t.name not in allow:
                            continue
                        full = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{name}__{t.name}")[:64]
                        self.route[full] = (name, t.name)
                        self.schemas.append({"type": "function", "function": {
                            "name": full,
                            "description": (t.description or "")[:500],
                            "parameters": t.inputSchema or {"type": "object", "properties": {}}}})
                    self.sessions[name] = session
                    self.cfg[name] = cfg
                    ready.set_result(True)
                    await stop.wait()
        except Exception as e:  # noqa: BLE001
            self.errors[name] = str(e)
            if not ready.done():
                ready.set_result(False)

    def start(self, servers: dict) -> None:
        for name, cfg in servers.items():
            ready = concurrent.futures.Future()
            asyncio.run_coroutine_threadsafe(self._serve(name, cfg, ready), self.loop)
            try:
                ready.result(timeout=90)
            except concurrent.futures.TimeoutError:
                self.errors[name] = "startup timeout"

    def call(self, full: str, args: dict) -> str:
        server, tool = self.route[full]
        if not (self.cfg[server].get("trust") or AUTO):
            if not confirm(f"MCP {server}.{tool} {json.dumps(args, ensure_ascii=False)[:200]}"):
                return "Denied by user"

        async def go():
            res = await self.sessions[server].call_tool(tool, args)
            parts = [getattr(c, "text", None) or f"[{c.type} content]" for c in res.content]
            return ("Error: " if res.isError else "") + "\n".join(parts)

        try:
            return truncate(asyncio.run_coroutine_threadsafe(go(), self.loop).result(timeout=120))
        except Exception as e:  # noqa: BLE001
            return f"Error: MCP call failed: {e}"

    def shutdown(self):
        for s in self.stops:
            self.loop.call_soon_threadsafe(s.set)
        time.sleep(0.5)


def load_mcp_config() -> dict:
    servers = {}
    for f in (HOME_CFG / "mcp.json", PROJ_CFG / "mcp.json"):
        if f.is_file():
            try:
                servers.update(json.loads(f.read_text()).get("servers", {}))
            except json.JSONDecodeError as e:
                print(f"[mcp] bad JSON in {f}: {e}")
    return servers


MCP = None


def init_mcp() -> None:
    global MCP
    servers = load_mcp_config()
    if not servers:
        return
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("[mcp] config found but package missing: pip install mcp")
        return
    MCP = MCPManager()
    print(f"[mcp] starting {len(servers)} server(s)...")
    MCP.start(servers)
    for name, err in MCP.errors.items():
        print(f"[mcp] {name} failed: {err}")


def all_tools() -> list:
    return BASE_TOOLS + (MCP.schemas if MCP else [])


def execute(name: str, args: dict) -> str:
    if MCP and name in MCP.route:
        return MCP.call(name, args)
    fn = IMPL.get(name)
    if not fn:
        return f"Error: unknown tool {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: bad arguments: {e}"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# =====================================================================
# prompt / project memory
# =====================================================================
def system_prompt() -> str:
    base = (
        "You are a coding agent working in a terminal inside the project directory.\n"
        "Inspect before changing: search, read, then edit. Make small, precise edits.\n"
        "After changes, run tests/linters if the project has them. Be concise. "
        "Answer in the user's language.\n"
        f"Project directory: {WORKDIR}\n"
    )
    for name in ("AGENTS.md", "CLAUDE.md"):
        f = WORKDIR / name
        if f.is_file():
            base += f"\n# Project instructions ({name})\n{truncate(f.read_text(), 6000)}\n"
            break
    if SKILLS:
        base += "\n# Available skills (call use_skill(name) when one matches the task)\n"
        base += "\n".join(f"- {n}: {s['desc']}" for n, s in SKILLS.items()) + "\n"
    return base


# =====================================================================
# session + context compaction
# =====================================================================
def session_file() -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if RESUME:
        existing = sorted(SESSION_DIR.glob("*.json"))
        if existing:
            return existing[-1]
    return SESSION_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"


def save(path: Path, messages: list) -> None:
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=1))


def history_size(messages: list) -> int:
    return len(json.dumps(messages, ensure_ascii=False))


def summarize(messages: list) -> bool:
    """Replace old turns with a model-written summary (cut only at a user message)."""
    cut = len(messages) - KEEP_RECENT
    while cut > 1 and messages[cut]["role"] != "user":
        cut -= 1
    if cut <= 1:
        return False
    lines = []
    for m in messages[1:cut]:
        body = m.get("content") or json.dumps(m.get("tool_calls", ""), ensure_ascii=False)
        lines.append(f"[{m['role']}] {truncate(str(body), 1200)}")
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.1,
        messages=[
            {"role": "system", "content":
                "Compress this coding-agent conversation into concise notes: user goal, "
                "decisions made, files read/changed (with paths), commands run and results, "
                "current state, open TODOs. Keep exact paths and identifiers. No fluff."},
            {"role": "user", "content": truncate("\n".join(lines), 40000)},
        ])
    summary = resp.choices[0].message.content or "(empty summary)"
    messages[1:cut] = [
        {"role": "user", "content": "[Summary of earlier conversation]\n" + summary},
        {"role": "assistant", "content": "Understood, continuing from the summary."},
    ]
    return True


def compact(messages: list, force: bool = False) -> None:
    if not force and history_size(messages) < TRIM_AT:
        return
    for m in messages[1:-KEEP_RECENT]:                      # cheap step: trim old tool output
        if m["role"] == "tool" and len(m["content"]) > 300:
            m["content"] = "[old tool output trimmed]"
    if force or history_size(messages) > SUMMARIZE_AT:      # expensive step: summarize
        print("\n  [compacting history...]")
        if summarize(messages):
            files_read.clear()      # old file contents are gone: require fresh reads before edits


# =====================================================================
# agent loop with streaming
# =====================================================================
def stream_completion(messages: list):
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, tools=all_tools(), temperature=0.2, stream=True)
    content, calls = [], {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content.append(delta.content)
        for tc in delta.tool_calls or []:
            key = tc.index if tc.index is not None else 0
            if tc.id and key in calls and calls[key]["id"] and calls[key]["id"] != tc.id:
                key = max(calls) + 1          # backend reused an index for a new call
            c = calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                c["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    c["name"] += tc.function.name
                if tc.function.arguments:
                    c["arguments"] += tc.function.arguments
    result = [calls[k] for k in sorted(calls)]
    for i, c in enumerate(result):
        c["id"] = c["id"] or f"call_{int(time.time())}_{i}"
    return "".join(content), result


def run_turn(messages: list, sfile: Path) -> None:
    for _ in range(MAX_STEPS):
        compact(messages)
        print()
        text, calls = stream_completion(messages)
        entry = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in calls]
        messages.append(entry)
        if not calls:
            print()
            save(sfile, messages)
            return
        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
                print(f"\n  > {c['name']} {json.dumps(args, ensure_ascii=False)[:150]}")
                result = execute(c["name"], args)
            except json.JSONDecodeError:
                result = "Error: arguments are not valid JSON"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": truncate(result)})
        save(sfile, messages)
    print("\n[step limit reached]")


def main() -> None:
    init_mcp()
    sfile = session_file()
    if RESUME and sfile.exists():
        messages = json.loads(sfile.read_text())
        messages[0] = {"role": "system", "content": system_prompt()}
        print(f"Resumed {sfile.name}")
    else:
        messages = [{"role": "system", "content": system_prompt()}]
    print(f"Agent | model={MODEL} | dir={WORKDIR} | {'AUTO' if AUTO else 'confirm mode'} | "
          f"skills={len(SKILLS)} | tools={len(all_tools())} | /compact /skills /mcp, 'exit' to quit")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"):
            break
        if user == "/skills":
            print("\n".join(f"- {n}: {s['desc']}" for n, s in SKILLS.items()) or "no skills")
            continue
        if user == "/mcp":
            print("\n".join(f"- {n}" for n in (MCP.route if MCP else [])) or "no MCP tools")
            continue
        if user == "/compact":
            compact(messages, force=True)
            save(sfile, messages)
            continue
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        try:
            run_turn(messages, sfile)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:  # noqa: BLE001
            print(f"\n[error] {e}")


if __name__ == "__main__":
    main()
