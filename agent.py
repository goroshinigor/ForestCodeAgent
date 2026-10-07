#!/usr/bin/env python3
"""Minimal terminal coding agent: Ollama (OpenAI-compatible) + tool loop.

Run:   python agent.py            (new session)
       python agent.py --resume   (continue last session)
       python agent.py --auto     (no confirmations: only inside a sandbox/VM!)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

MODEL = os.environ.get("AGENT_MODEL", "coder-local")
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:11434/v1")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", os.getcwd())).resolve()
SESSION_DIR = WORKDIR / ".agent" / "sessions"
MAX_OUTPUT = 8000          # chars per tool result
MAX_STEPS = 30             # tool-loop iterations per user message
COMPACT_AT = 70000         # chars of history before old tool results get trimmed
KEEP_RECENT = 8            # messages never trimmed

AUTO = "--auto" in sys.argv
RESUME = "--resume" in sys.argv

client = OpenAI(base_url=BASE_URL, api_key="ollama")
files_read: set[str] = set()

SAFE_CMD = re.compile(
    r"^(ls|pwd|cat|head|tail|wc|rg|grep|find|git (status|diff|log|branch)|php -v|python3? --version)\b"
)
SHELL_META = re.compile(r"[;&|><`]|\$\(")


# ---------- helpers ----------
def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    """Обрезает текст до заданного лимита, оставляя начало и конец.
    
    Args:
        text: Текст для обрезки
        limit: Максимальное количество символов (по умолчанию MAX_OUTPUT)
        
    Returns:
        Обрезанный текст с уведомлением об обрезке
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-half:]


def safe_path(p: str) -> Path:
    full = (WORKDIR / p).resolve()
    if full != WORKDIR and WORKDIR not in full.parents:
        raise ValueError(f"path outside workdir: {p}")
    return full


def confirm(prompt: str) -> bool:
    if AUTO:
        return True
    return input(f"\n  ? {prompt} [y/N] ").strip().lower() in ("y", "yes", "д", "да")


# ---------- tools ----------
def read_file(path: str, offset: int = 0, limit: int = 400) -> str:
    p = safe_path(path)
    if not p.is_file():
        return f"Error: {path} is not a file"
    files_read.add(str(p))
    lines = p.read_text(errors="replace").splitlines()
    chunk = lines[offset: offset + limit]
    out = "\n".join(chunk)
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
TOOLS = [
    _schema("read_file", "Read a text file (400 lines per call).", {"path": S, "offset": I, "limit": I}, ["path"]),
    _schema("write_file", "Create a new file or overwrite one you have read.", {"path": S, "content": S}, ["path", "content"]),
    _schema("edit_file", "Replace a UNIQUE exact string in a file you have read.", {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    _schema("list_files", "List files by glob pattern, e.g. 'src/**/*.php'.", {"pattern": S}, []),
    _schema("grep", "Search file contents by regex.", {"pattern": S, "path": S}, ["pattern"]),
    _schema("bash", "Run a shell command in the project dir.", {"command": S, "timeout": I}, ["command"]),
]
IMPL = {f.__name__: f for f in (read_file, write_file, edit_file, list_files, grep, bash)}


def execute(name: str, args: dict) -> str:
    fn = IMPL.get(name)
    if not fn:
        return f"Error: unknown tool {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: bad arguments: {e}"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# ---------- prompt / memory ----------
def system_prompt() -> str:
    base = (
        "You are a coding agent working in a terminal inside the project directory.\n"
        "Use tools to inspect before changing: search, read, then edit. Make small, precise edits.\n"
        "After changes, run tests/linters if the project has them. Be concise. "
        "Answer in the user's language.\n"
        f"Project directory: {WORKDIR}\n"
    )
    for name in ("AGENTS.md", "CLAUDE.md"):
        f = WORKDIR / name
        if f.is_file():
            base += f"\n# Project instructions ({name})\n{truncate(f.read_text(), 6000)}\n"
            break
    return base


# ---------- session ----------
def session_file() -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if RESUME:
        existing = sorted(SESSION_DIR.glob("*.json"))
        if existing:
            return existing[-1]
    return SESSION_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"


def save(path: Path, messages: list) -> None:
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=1))


def compact(messages: list) -> None:
    """Crude context control: trim old tool results once history grows large."""
    if len(json.dumps(messages)) < COMPACT_AT:
        return
    for m in messages[1:-KEEP_RECENT]:
        if m["role"] == "tool" and len(m["content"]) > 300:
            m["content"] = "[old tool output trimmed]"


# ---------- loop ----------
def run_turn(messages: list, sfile: Path) -> None:
    for _ in range(MAX_STEPS):
        compact(messages)
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0.2)
        msg = resp.choices[0].message
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]
        messages.append(entry)
        if msg.content:
            print(f"\n{msg.content}")
        if not msg.tool_calls:
            save(sfile, messages)
            return
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                print(f"\n  > {tc.function.name} {json.dumps(args, ensure_ascii=False)[:150]}")
                result = execute(tc.function.name, args)
            except json.JSONDecodeError:
                result = "Error: arguments are not valid JSON"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate(result)})
        save(sfile, messages)
    print("\n[step limit reached]")


def main() -> None:
    sfile = session_file()
    if RESUME and sfile.exists():
        messages = json.loads(sfile.read_text())
        messages[0] = {"role": "system", "content": system_prompt()}
        print(f"Resumed {sfile.name}")
    else:
        messages = [{"role": "system", "content": system_prompt()}]
    print(f"Agent | model={MODEL} | dir={WORKDIR} | {'AUTO' if AUTO else 'confirm mode'} | 'exit' to quit")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit"):
            break
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
