#!/usr/bin/env python3
"""
CivicMapper ETL AI Orchestrator.

Reads a city contribution Markdown file, calls the Claude API with an
agentic tool-use loop, outputs the generated ETL script, and (when
DRY_RUN=false) executes it with a 1-hour timeout.

During generation Claude can call the `execute_python` tool to probe the
city's data source, inspect actual field names, and test transformation
logic before committing to the final script.

If execution fails, automatically retries up to MAX_RETRIES times. Claude
may return a corrected script (possibly via tools) or a MARKDOWN_ISSUE
block if the city Markdown itself contains incorrect information.

Usage:
    python etl_ai_orchestrator.py --city-file /path/to/city.md --output-dir /path/to/output

Output:
    {output_dir}/generated_etl.py            — the final (working) ETL script
    {output_dir}/generated_etl_attempt{N}.py — scripts from retry attempts
    {output_dir}/orchestrator_result.json    — structured result for CI reporting

Environment variables:
    CLAUDE_API_KEY      — required
    PROMPTS_DIR         — optional, defaults to data/prompts/
    DRY_RUN             — set to "false" to enable ETL execution (Phase 2+)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import anthropic

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "data/prompts"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"

# Maximum number of full execution retry attempts after the initial generation.
# Total attempts = 1 (initial) + MAX_RETRIES.
MAX_RETRIES = 3

# Maximum tool calls per agentic generation round (initial or retry).
# Keep this low — Claude should be targeted (probe fields, test one sample,
# verify output) rather than exhaustively exploring.
MAX_TOOL_CALLS = 8

# Timeout (seconds) for each execute_python snippet during the tool-use loop.
SNIPPET_TIMEOUT = 90

# Maximum characters of a tool result stored back into the conversation.
# Keeps the growing context from blowing through the per-minute token limit.
TOOL_RESULT_MAX_CHARS = 2000


# ── API call with rate-limit backoff ──────────────────────────────────────────

def _create_with_backoff(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    """Call client.messages.create, retrying on 429 RateLimitError with backoff."""
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            if attempt == max_retries:
                raise
            # Respect retry-after header if present, otherwise exponential backoff
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after = exc.response.headers.get("retry-after")
            wait = int(retry_after) if retry_after else (30 * (2 ** attempt))
            print(f"  ⏳ Rate limited — waiting {wait}s (retry {attempt + 1}/{max_retries})…")
            time.sleep(wait)


# ── Tool definitions ──────────────────────────────────────────────────────────

def make_tools() -> list[dict]:
    """Return the tool definitions to pass to every Claude API call."""
    return [
        {
            "name": "execute_python",
            "description": (
                "Run a Python code snippet and return its stdout and stderr. "
                "Use this to probe the city's data source (ArcGIS endpoint, open data URL), "
                "inspect actual field names and values, test field mappings, "
                "validate CRS, or verify transformation logic with a small sample "
                "before writing the final script. "
                "Available packages: requests, geopandas, pandas, pyarrow, shapely, osmnx. "
                f"Timeout: {SNIPPET_TIMEOUT} seconds. "
                "Do NOT upload files to Azure Blob Storage during exploration — "
                "save that for the final script."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to show results.",
                    }
                },
                "required": ["code"],
            },
        }
    ]


def _run_python_snippet(code: str, sandbox_dir: Path) -> str:
    """Execute a Python snippet safely and return a combined result string."""
    snippet_path = sandbox_dir / "snippet.py"
    snippet_path.write_text(code, encoding="utf-8")

    # Pass environment but strip Azure credentials — exploration only.
    env = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_")}
    env["PYTHONUNBUFFERED"] = "1"
    env["SCRAPE_DATA"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, str(snippet_path)],
            capture_output=True,
            text=True,
            timeout=SNIPPET_TIMEOUT,
            cwd=str(sandbox_dir),
            env=env,
        )
        stdout = (proc.stdout or "")[-3000:]
        stderr = (proc.stderr or "")[-2000:]
        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        parts.append(f"Exit code: {proc.returncode}")
        return "\n\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Timed out after {SNIPPET_TIMEOUT} seconds."
    except Exception as exc:
        return f"Error running snippet: {exc}"


def _dispatch_tool(name: str, tool_input: dict, sandbox_dir: Path) -> str:
    """Execute a tool call and return its string result."""
    if name == "execute_python":
        return _run_python_snippet(tool_input["code"], sandbox_dir)
    return f"Unknown tool: {name}"


# ── Core agentic loop ─────────────────────────────────────────────────────────

def _run_agentic_loop(
    client: anthropic.Anthropic,
    bundle: dict,
    messages: list[dict],
    sandbox_dir: Path,
) -> tuple[str, dict, int]:
    """Run the Claude tool-use loop until end_turn or MAX_TOOL_CALLS.

    Returns (final_text, usage_totals, tool_call_count).
    final_text is the last assistant text block — callers extract the script from it.
    """
    tools = make_tools()
    total_input_tokens = 0
    total_output_tokens = 0
    tool_call_count = 0
    final_text = ""

    while True:
        response = _create_with_backoff(
            client,
            model=bundle["model"],
            max_tokens=8192,
            system=bundle["system_prompt"],
            tools=tools,
            messages=messages,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Collect any text from this response (there may be a text block before tool calls)
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text  # keep overwriting — we want the last one

        # Append the full assistant response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            # Unexpected — stop and return whatever text we have
            print(f"⚠️  Unexpected stop_reason: {response.stop_reason}")
            break

        # Execute tool calls and collect results.
        # Cap result content stored in the conversation to keep context size bounded.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_call_count += 1
                first_line = block.input.get("code", "").splitlines()[0][:80] if block.name == "execute_python" else ""
                print(f"  🔧 Tool call {tool_call_count}/{MAX_TOOL_CALLS}: {block.name}"
                      + (f" — {first_line}…" if first_line else ""))

                result = _dispatch_tool(block.name, block.input, sandbox_dir)

                # Truncate what goes back into the conversation to keep input tokens bounded
                stored = (result[:TOOL_RESULT_MAX_CHARS] + "\n…[truncated]"
                          if len(result) > TOOL_RESULT_MAX_CHARS else result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": stored,
                })

        messages.append({"role": "user", "content": tool_results})

        # If we've hit the cap, ask Claude to conclude on the next turn
        if tool_call_count >= MAX_TOOL_CALLS:
            print(f"  ⚠️  Reached {MAX_TOOL_CALLS} tool calls — asking Claude to conclude.")
            messages.append({
                "role": "user",
                "content": (
                    f"You have used {MAX_TOOL_CALLS} tool calls. "
                    "Please output the complete final Python ETL script now based on "
                    "what you have learned. Return ONLY the script, no prose or fences."
                ),
            })
            # One final response with no tools offered
            final_response = _create_with_backoff(
                client,
                model=bundle["model"],
                max_tokens=8192,
                system=bundle["system_prompt"],
                messages=messages,
            )
            total_input_tokens += final_response.usage.input_tokens
            total_output_tokens += final_response.usage.output_tokens
            for block in final_response.content:
                if hasattr(block, "text"):
                    final_text = block.text
            break

    usage = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "model": bundle["model"],
        "prompt_version": bundle["version"],
        "tool_calls": tool_call_count,
    }
    return final_text, usage, tool_call_count


# ── Public generation functions ───────────────────────────────────────────────

def load_prompt_bundle(version: str) -> dict:
    """Load all files from the versioned prompt bundle."""
    bundle_dir = PROMPTS_DIR / version
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Prompt bundle not found: {bundle_dir}")

    def read(name: str) -> str:
        p = bundle_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    few_shot_dir = bundle_dir / "few_shot"
    few_shots = {}
    if few_shot_dir.exists():
        for f in sorted(few_shot_dir.glob("*.py")):
            few_shots[f.stem] = f.read_text(encoding="utf-8")

    return {
        "version": version,
        "system_prompt": read("system_prompt.txt"),
        "playbook": read("playbook.md"),
        "schema": read("schema.md"),
        "model": read("model.txt").strip() or "claude-sonnet-4-6",
        "few_shots": few_shots,
    }


def parse_city_metadata(markdown: str) -> dict:
    """Extract key city metadata from the contribution markdown for logging."""
    meta = {}

    m = re.search(r'\*\*City key\*\*\s*\|\s*`([^`]+)`', markdown)
    if m:
        meta["city_key"] = m.group(1).strip()

    m = re.search(r'\*\*State code\*\*\s*\|\s*`([^`]+)`', markdown)
    if m:
        meta["state"] = m.group(1).strip()

    m = re.search(r'\*\*Display name\*\*\s*\|\s*`?([^`|\n]+)`?', markdown)
    if m:
        meta["display_name"] = m.group(1).strip().strip("`")

    m = re.search(r'\*\*Approx\. parcel count\*\*\s*\|\s*([^\n|]+)', markdown)
    if m:
        meta["approx_parcels"] = m.group(1).strip()

    m = re.search(r'\*\*PMTiles recommended\*\*\s*\|\s*([^\n|]+)', markdown)
    if m:
        meta["pmtiles_recommended"] = "yes" in m.group(1).lower()

    m = re.search(r'\*\*Include parking dataset\?\*\*\s*\|\s*([^\n|]+)', markdown)
    if m:
        meta["include_parking"] = "yes" in m.group(1).lower()

    return meta


def build_user_message(bundle: dict, city_markdown: str) -> str:
    """Construct the full user message with all context."""
    few_shot_section = ""
    for city_name, code in bundle["few_shots"].items():
        few_shot_section += f"\n\n### Reference ETL: {city_name}.py\n```python\n{code}\n```"

    return textwrap.dedent(f"""
        ## Task

        Generate a complete Python ETL script for the city described below.

        You have access to an `execute_python` tool. Use it to verify your approach
        before writing the final script. Specifically:
        - Probe the data source URL to confirm the actual field names (they often differ
          from what is described in the city Markdown — always verify)
        - Fetch a small sample (e.g., first 100 features) to check geometry types and CRS
        - Test your field mapping and any transformation logic before committing

        When you are satisfied with your testing, output the complete final Python ETL
        script as your LAST message. Return ONLY the script — no prose, no markdown fences.
        The script must be ready to run from the `data/jurisidictions/` directory.

        ## Schema Reference

        {bundle["schema"]}

        ## Playbook

        {bundle["playbook"]}

        ## Reference ETL Scripts (few-shot examples)
        {few_shot_section}

        ## City Contribution Markdown

        ```markdown
        {city_markdown}
        ```
    """).strip()


def _strip_fences(raw: str) -> str:
    """Remove any accidental markdown code fences Claude may have added."""
    raw = re.sub(r'^```python\s*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```\s*$', '', raw, flags=re.MULTILINE)
    return raw.strip()


def _parse_correction_notes(script: str) -> list[str]:
    """Extract lines from a # CORRECTION_NOTES: block at the top of the script."""
    notes = []
    in_notes = False
    for line in script.splitlines():
        stripped = line.strip()
        if stripped == "# CORRECTION_NOTES:":
            in_notes = True
        elif in_notes and stripped.startswith("# - "):
            notes.append(stripped[4:].strip())
        elif in_notes and stripped.startswith("#"):
            continue  # other comment line inside the block
        elif in_notes:
            break  # first non-comment line ends the block
    return notes


def call_claude(bundle: dict, city_markdown: str, sandbox_dir: Path) -> tuple[str, dict]:
    """Run the agentic generation loop and return (final_script, usage_stats)."""
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    user_message = build_user_message(bundle, city_markdown)
    messages = [{"role": "user", "content": user_message}]

    print(f"🤖 Starting agentic ETL generation "
          f"(model={bundle['model']}, prompt_version={bundle['version']}, "
          f"max_tool_calls={MAX_TOOL_CALLS})...")

    final_text, usage, tool_call_count = _run_agentic_loop(
        client, bundle, messages, sandbox_dir
    )

    script = _strip_fences(final_text)
    print(f"✅ Script generated: {len(script):,} chars, "
          f"{tool_call_count} tool calls, {usage['output_tokens']:,} output tokens")
    return script, usage


def call_claude_with_feedback(
    bundle: dict,
    city_markdown: str,
    failed_script: str,
    error_info: dict,
    attempt_num: int,
    sandbox_dir: Path,
) -> dict:
    """Send an error-feedback round to Claude (with tools) and return a correction result.

    Returns a dict:
        {
          "script": str | None,          # corrected script, or None if markdown issue
          "is_markdown_issue": bool,
          "issue_explanation": str | None,
          "notes": list[str],            # from # CORRECTION_NOTES: block
          "usage": dict,
        }
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    user_message = build_user_message(bundle, city_markdown)

    stderr_snippet = (error_info.get("stderr") or "(no stderr)")[-2000:]
    exit_code = error_info.get("exit_code", "unknown")

    feedback = textwrap.dedent(f"""
        The script you generated failed during execution (attempt {attempt_num - 1} of {MAX_RETRIES + 1}).

        **Error details**
        Exit code: {exit_code}

        Last stderr output:
        ```
        {stderr_snippet}
        ```

        You may use the `execute_python` tool to test your fix before returning the
        corrected script.

        **IMPORTANT — logic deviation rule:**
        If fixing the error requires changing core logic described in the city Markdown —
        such as field mappings, exemption logic, category logic, or the CRS — because the
        Markdown itself appears to contain incorrect information, do NOT silently make that
        change. Instead, return ONLY a block starting exactly with:

            MARKDOWN_ISSUE:
            <clear explanation of what in the Markdown seems incorrect and what the correct
            value appears to be based on the actual data source>

        If the fix is purely a coding/implementation change (different API pagination,
        different library call, different error handling, etc.) you may make it freely.

        **If returning a corrected script**, optionally prefix it with a comment block
        (immediately before any imports) listing what you changed:

            # CORRECTION_NOTES:
            # - Brief note about what you fixed
            # - Another note if applicable

        When you are done, return ONLY the corrected Python script OR the MARKDOWN_ISSUE
        block. No prose, no markdown fences.
    """).strip()

    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": failed_script},
        {"role": "user",      "content": feedback},
    ]

    print(f"🔄 Retry attempt {attempt_num} — running agentic correction loop...")
    final_text, usage, tool_call_count = _run_agentic_loop(
        client, bundle, messages, sandbox_dir
    )

    raw = _strip_fences(final_text)

    # Detect MARKDOWN_ISSUE response
    if raw.startswith("MARKDOWN_ISSUE:"):
        issue_text = raw[len("MARKDOWN_ISSUE:"):].strip()
        print(f"⚠️  Claude flagged a markdown issue: {issue_text[:200]}")
        return {
            "script": None,
            "is_markdown_issue": True,
            "issue_explanation": issue_text,
            "notes": [],
            "usage": usage,
        }

    notes = _parse_correction_notes(raw)
    return {
        "script": raw,
        "is_markdown_issue": False,
        "issue_explanation": None,
        "notes": notes,
        "usage": usage,
    }


# ── ETL execution ─────────────────────────────────────────────────────────────

def execute_etl(script_path: Path, city_meta: dict, output_dir: Path) -> dict:
    """Execute the generated ETL script with a 1-hour timeout."""
    run_dir = Path("data/jurisidictions")
    if not run_dir.exists():
        run_dir = Path(".")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CI": "true",
        "SCRAPE_DATA": "1",
        "ETL_OUTPUT_DIR": str(output_dir.resolve()),
    }

    print(f"🚀 Executing ETL script: {script_path}")
    print(f"   Working dir: {run_dir.resolve()}")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path.resolve())],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(run_dir),
            env=env,
        )
        execution_result = {
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
            "stdout": proc.stdout[-5000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "timed_out": False,
        }
        status = "✅" if proc.returncode == 0 else "❌"
        print(f"{status} ETL exit code: {proc.returncode}")
        if proc.stderr:
            print("STDERR (last 2000 chars):\n" + proc.stderr[-2000:])
    except subprocess.TimeoutExpired as exc:
        execution_result = {
            "exit_code": -1,
            "success": False,
            "stdout": exc.stdout[-5000:] if exc.stdout else "",
            "stderr": exc.stderr[-2000:] if exc.stderr else "",
            "timed_out": True,
        }
        print("❌ ETL execution timed out after 1 hour")

    return execution_result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CivicMapper ETL AI Orchestrator")
    parser.add_argument("--city-file",  required=True, help="Path to the city contribution .md file")
    parser.add_argument("--output-dir", required=True, help="Directory to write output artifacts")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                        help="Generate script only — do not execute")
    args = parser.parse_args()

    city_path = Path(args.city_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sandbox directory for tool-use snippet execution
    sandbox_dir = output_dir / "sandbox"
    sandbox_dir.mkdir(exist_ok=True)

    if not city_path.exists():
        print(f"❌ City file not found: {city_path}")
        sys.exit(1)

    city_markdown = city_path.read_text(encoding="utf-8")
    city_meta = parse_city_metadata(city_markdown)
    print(f"📍 City: {city_meta.get('display_name', city_path.stem)}  "
          f"({city_meta.get('city_key', '?')}, {city_meta.get('state', '?')})")

    version_file = PROMPTS_DIR / "CURRENT_VERSION"
    if not version_file.exists():
        print(f"❌ CURRENT_VERSION not found at {version_file}")
        sys.exit(1)
    version = version_file.read_text(encoding="utf-8").strip()
    bundle = load_prompt_bundle(version)
    print(f"📦 Prompt bundle: {version}  model: {bundle['model']}")

    # ── Initial agentic generation ────────────────────────────────────────────
    generated_script, usage = call_claude(bundle, city_markdown, sandbox_dir)

    script_path = output_dir / "generated_etl.py"
    script_path.write_text(generated_script, encoding="utf-8")
    print(f"✅ Saved generated_etl.py  ({len(generated_script):,} chars)")

    if args.dry_run:
        print("ℹ️  DRY_RUN=true — script generated but not executed.")
        result = {
            "status": "success",
            "dry_run": True,
            "city_metadata": city_meta,
            "prompt_bundle_version": version,
            "model": bundle["model"],
            "generated_script_path": str(script_path),
            "generated_script_lines": len(generated_script.splitlines()),
            "api_usage": usage,
            "execution": None,
            "total_attempts": 1,
            "attempts": [{"attempt": 1, "usage": usage, "execution": None}],
            "notes": [],
            "markdown_issue": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        result_path = output_dir / "orchestrator_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"✅ Result JSON: {result_path}")
        return result

    # ── Execute (with retry loop) ─────────────────────────────────────────────
    execution_result = execute_etl(script_path, city_meta, output_dir)

    attempts = [{"attempt": 1, "usage": usage, "execution": execution_result}]
    notes: list[str] = []
    markdown_issue: str | None = None
    current_script = generated_script

    for retry_num in range(1, MAX_RETRIES + 1):
        if execution_result["success"]:
            break
        if execution_result.get("timed_out"):
            print("⏱️  Timed out — skipping retries (not a code issue).")
            break

        attempt_num = retry_num + 1
        print(f"\n{'─' * 60}")
        print(f"Attempt {attempt_num} of {MAX_RETRIES + 1}")

        retry = call_claude_with_feedback(
            bundle, city_markdown, current_script, execution_result, attempt_num, sandbox_dir
        )

        if retry["is_markdown_issue"]:
            markdown_issue = retry["issue_explanation"]
            attempts.append({
                "attempt": attempt_num,
                "usage": retry["usage"],
                "execution": None,
                "markdown_issue": markdown_issue,
            })
            break

        notes.extend(retry["notes"])
        current_script = retry["script"]

        attempt_path = output_dir / f"generated_etl_attempt{attempt_num}.py"
        attempt_path.write_text(current_script, encoding="utf-8")

        execution_result = execute_etl(attempt_path, city_meta, output_dir)
        attempts.append({
            "attempt": attempt_num,
            "usage": retry["usage"],
            "execution": execution_result,
        })

    # Overwrite generated_etl.py with the final winning script
    if execution_result and execution_result["success"] and current_script != generated_script:
        script_path.write_text(current_script, encoding="utf-8")
        print(f"✅ Overwrote generated_etl.py with successful attempt script")

    # ── Determine overall status ──────────────────────────────────────────────
    if markdown_issue:
        overall_status = "markdown_issue"
    elif execution_result and execution_result["success"]:
        overall_status = "success"
    else:
        overall_status = "etl_failed"

    # ── Write structured result ───────────────────────────────────────────────
    result = {
        "status": overall_status,
        "dry_run": False,
        "city_metadata": city_meta,
        "prompt_bundle_version": version,
        "model": bundle["model"],
        "generated_script_path": str(script_path),
        "generated_script_lines": len(current_script.splitlines()),
        "api_usage": usage,
        "execution": execution_result,
        "total_attempts": len(attempts),
        "attempts": attempts,
        "notes": notes,
        "markdown_issue": markdown_issue,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    result_path = output_dir / "orchestrator_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✅ Result JSON: {result_path}")

    if overall_status != "success":
        sys.exit(1)

    return result


if __name__ == "__main__":
    main()
