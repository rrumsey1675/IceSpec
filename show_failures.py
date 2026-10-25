#!/usr/bin/env python3

# show_failures.py
import csv
import sys
import textwrap
from pathlib import Path

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MAX_ERROR_LINES = 13       # limit to first few lines of stderr
WRAP_WIDTH = 90           # terminal line width for wrapping text

# ANSI colors (for pretty output)
RED   = "\033[31m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELL  = "\033[33m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ────────────────────────────────────────────────
# Functions
# ────────────────────────────────────────────────
def wrap_text(text: str, width: int = WRAP_WIDTH, indent: int = 4) -> str:
    """Wraps and indents stderr excerpts for readability."""
    if not text.strip():
        return " " * indent + "(no stderr captured)"
    lines = text.strip().splitlines()
    if len(lines) > MAX_ERROR_LINES:
        lines = lines[:MAX_ERROR_LINES] + ["... (truncated) ..."]
    wrapped = "\n".join(
        textwrap.fill(line, width=width, subsequent_indent=" " * indent)
        for line in lines
    )
    return " " * indent + wrapped.replace("\n", "\n" + " " * indent)


def show_failures(log_path: Path) -> None:
    """Reads a batch log CSV and prints a readable failure summary."""
    if not log_path.exists():
        print(f"{RED}[failures]{RESET} Log not found: {log_path}")
        return

    with log_path.open() as f:
        reader = csv.DictReader(f)
        fails = [row for row in reader if row["returncode"] != "0"]

    if not fails:
        print(f"{GREEN}[failures] 🎉 No failures found!{RESET}")
        return

    print(f"{YELL}{BOLD}Found {len(fails)} failure(s):{RESET}\n")

    for idx, row in enumerate(fails, start=1):
        print(f"{BOLD}{idx}. {CYAN}{row['image_id']}{RESET}")
        print(f"   Return code: {RED}{row['returncode']}{RESET}")
        print(f"   Duration: {row['duration_s']} s")
        if row.get("output_dir"):
            print(f"   Output dir: {row['output_dir']}")
        print(wrap_text(row.get("stderr_excerpt", "")))
        print()  # blank line between entries


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Pass path as first argument or default to most recent log
    log_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    log_path = Path(log_arg) if log_arg else None

    if log_path is None or not log_path.exists():
        # Optionally: auto-pick latest log folder if none provided
        root = Path("/data/icecores/batch_logs")
        latest_dir = max(root.iterdir(), key=lambda d: d.stat().st_mtime)
        log_path = latest_dir / "batch_log.csv"
        print(f"[failures] No path given — using latest log: {log_path}")

    show_failures(log_path)