"""Claude Code PostToolUse hook: run ruff and pytest after a code file is edited.

Claude Code starts this script after every Edit or Write and passes a JSON description
of the tool call on stdin. If the edited file is Python code or ``pyproject.toml``, the
script runs the linter and the test suite. On failure it prints the tool output to
stderr and exits with code 2, which is how a PostToolUse hook hands a message back to
the agent. On success, and for any other file, it stays silent.

The formatter is not run here on purpose: rewriting a file right after the agent
edited it makes the agent's next edit fail on stale text.
"""

import json
import subprocess
import sys
from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKED_SUFFIXES = frozenset({".py"})
CHECKED_FILE_NAMES = frozenset({"pyproject.toml"})
EXIT_OK = 0
EXIT_REPORT_TO_AGENT = 2
# Claude Code cuts hook output off at 10,000 characters; keep the end, where ruff and
# pytest print their summaries.
MAX_REPORT_CHARS = 6000
CHECKS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "."),
    ("pytest", "-q"),
)


def needs_check(file_path: str) -> bool:
    """Decide whether an edit to this file can break lint or tests.

    Args:
        file_path: Path of the edited file as Claude Code reports it; on Windows it
            uses backslashes.

    Returns:
        True for Python files and ``pyproject.toml``, False for everything else.
    """
    # PureWindowsPath splits on both "/" and "\", so one parser serves every platform.
    path = PureWindowsPath(file_path)
    return path.suffix.lower() in CHECKED_SUFFIXES or path.name.lower() in CHECKED_FILE_NAMES


def edited_file(hook_input: str) -> str:
    """Read the edited file's path out of the hook's stdin payload.

    Args:
        hook_input: JSON text Claude Code wrote to stdin.

    Returns:
        The value of ``tool_input.file_path``, or an empty string if the payload is
        not JSON or has no such field.
    """
    try:
        payload = json.loads(hook_input)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return ""
    file_path = tool_input.get("file_path")
    return file_path if isinstance(file_path, str) else ""


def run_checks() -> list[str]:
    """Run ruff and pytest in the project root.

    Returns:
        One report per failed check (command line plus the end of its output); empty
        if every check passed.
    """
    failures: list[str] = []
    for check in CHECKS:
        completed = subprocess.run(
            [sys.executable, "-m", *check],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            output = (completed.stdout + completed.stderr).strip()
            failures.append(f"$ {' '.join(check)}\n{output[-MAX_REPORT_CHARS:]}")
    return failures


def main() -> int:
    """Run the hook.

    Returns:
        0 if nothing had to be checked or every check passed, 2 if a check failed.
    """
    hook_input = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    if not needs_check(edited_file(hook_input)):
        return EXIT_OK
    failures = run_checks()
    if not failures:
        return EXIT_OK
    report = "Post-edit check failed. Fix this before going on.\n\n" + "\n\n".join(failures)
    sys.stderr.buffer.write(report.encode("utf-8", errors="replace"))
    return EXIT_REPORT_TO_AGENT


if __name__ == "__main__":
    sys.exit(main())
