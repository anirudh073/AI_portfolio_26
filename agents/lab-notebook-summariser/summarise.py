#!/usr/bin/env python3
"""
Lab notebook summarisation agent.

Usage:
    python agent/summarise.py week <month_dir> <week_dir>
    python agent/summarise.py month <month_dir>
    python agent/summarise.py all [--list-skipped]

Examples:
    python agent/summarise.py week nov_2025 week_2
    python agent/summarise.py month nov_2025
    python agent/summarise.py all
    python agent/summarise.py all --list-skipped
"""

import sys
import requests
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CHRONICLE_DIR = REPO_ROOT / "chronicle"
SUMMARIES_DIR = REPO_ROOT / "summaries"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

WEEK_PROMPT = """\
You are summarising a researcher's lab notebook entries for a single week.
The entries below are from different days. Write a concise, well-structured
weekly summary in markdown covering:
- What was accomplished
- Key findings or results
- Problems encountered and how they were addressed
- Outstanding to-dos

Be specific — preserve important technical details (tool names, dataset IDs,
analysis methods). Do not add information not present in the notes.

--- LAB ENTRIES ---
{content}
--- END ---
"""

MONTH_PROMPT = """\
You are summarising a researcher's lab notebook entries for an entire month.
The entries below span multiple weeks. Write a concise, well-structured
monthly summary in markdown covering:
- Major accomplishments and milestones
- Key findings or results
- Recurring problems and their solutions
- Ongoing threads and outstanding to-dos

Group related topics together. Be specific — preserve important technical
details. Do not add information not present in the notes.

--- LAB ENTRIES ---
{content}
--- END ---
"""

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def _assert_inside(path: Path, root: Path, label: str) -> None:
    """Raise PermissionError if path is not inside root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise PermissionError(
            f"{label} path '{path}' is outside the allowed directory '{root}'."
        )

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def read_chronicle_files(paths: list[Path]) -> str:
    """Read and concatenate .md files, prefixed with day headers."""
    parts = []
    for p in sorted(paths):
        _assert_inside(p, CHRONICLE_DIR, "Source")
        text = p.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"### {p.name}\n{text}")
    if not parts:
        sys.exit("Error: All matched files were empty.")
    return "\n\n---\n\n".join(parts)


def call_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return the response text."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        sys.exit("Error: Cannot reach Ollama. Is it running? Try: ollama serve")
    except requests.exceptions.Timeout:
        sys.exit("Error: Ollama request timed out (>5 min).")
    except (KeyError, ValueError):
        sys.exit(f"Error: Unexpected Ollama response:\n{resp.text}")


def write_summary(output_path: Path, content: str) -> None:
    """Write summary to output_path. Never overwrites existing files."""
    _assert_inside(output_path, SUMMARIES_DIR, "Output")
    if output_path.exists():
        sys.exit(
            f"Error: Summary already exists at '{output_path.relative_to(REPO_ROOT)}'.\n"
            "Delete or rename it manually if you want to regenerate."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Done. Summary written to: {output_path.relative_to(REPO_ROOT)}")


def already_summarised(output_path: Path) -> bool:
    return output_path.exists()


_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def month_has_ended(month_dir_name: str) -> bool:
    """Return True if the calendar month has fully passed as of today."""
    parts = month_dir_name.split("_")
    if len(parts) != 2:
        return True
    month_abbr, year_str = parts
    month_num = _MONTH_ABBR.get(month_abbr.lower())
    if month_num is None or not year_str.isdigit():
        return True
    year = int(year_str)
    today = date.today()
    return (today.year, today.month) > (year, month_num)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def summarise_week(month_dir: str, week_dir: str, skip_existing: bool = False) -> bool:
    """Summarise one week. Returns False if skipped, True if written."""
    output_path = SUMMARIES_DIR / "weekly" / f"{month_dir}_{week_dir}.md"
    if skip_existing and already_summarised(output_path):
        return False

    source_dir = CHRONICLE_DIR / month_dir / week_dir
    _assert_inside(source_dir, CHRONICLE_DIR, "Source")

    if not source_dir.is_dir():
        sys.exit(f"Error: Not found: chronicle/{month_dir}/{week_dir}")

    md_files = list(source_dir.glob("*.md"))
    if not md_files:
        sys.exit(f"Error: No .md files in chronicle/{month_dir}/{week_dir}")

    print(f"Reading {len(md_files)} file(s) from {month_dir}/{week_dir} ...")
    content = read_chronicle_files(md_files)

    print(f"Calling {MODEL} ...")
    summary = call_ollama(WEEK_PROMPT.format(content=content))

    header = f"# Weekly Summary — {month_dir} / {week_dir}\n\n"
    write_summary(output_path, header + summary)
    return True


def summarise_month(month_dir: str, skip_existing: bool = False) -> bool:
    """Summarise one month. Returns False if skipped, True if written."""
    output_path = SUMMARIES_DIR / "monthly" / f"{month_dir}.md"
    if skip_existing and already_summarised(output_path):
        return False

    source_dir = CHRONICLE_DIR / month_dir
    _assert_inside(source_dir, CHRONICLE_DIR, "Source")

    if not source_dir.is_dir():
        sys.exit(f"Error: Not found: chronicle/{month_dir}")

    md_files = list(source_dir.rglob("*.md"))
    if not md_files:
        sys.exit(f"Error: No .md files under chronicle/{month_dir}")

    print(f"Reading {len(md_files)} file(s) from {month_dir} ...")
    content = read_chronicle_files(md_files)

    print(f"Calling {MODEL} ...")
    summary = call_ollama(MONTH_PROMPT.format(content=content))

    header = f"# Monthly Summary — {month_dir}\n\n"
    write_summary(output_path, header + summary)
    return True


def summarise_all(list_skipped: bool = False) -> None:
    """Summarise every week and month found in chronicle/, skipping existing summaries."""
    if not CHRONICLE_DIR.is_dir():
        sys.exit("Error: chronicle/ directory not found.")

    skipped = []
    weeks_done = months_done = 0

    # Weekly summaries
    for week_dir in sorted(CHRONICLE_DIR.rglob("week_*")):
        if not week_dir.is_dir():
            continue
        if not list(week_dir.glob("*.md")):
            continue  # empty week directory — nothing to summarise
        month_dir = week_dir.parent.name
        week_name = week_dir.name
        written = summarise_week(month_dir, week_name, skip_existing=True)
        if written:
            weeks_done += 1
        else:
            skipped.append(f"weekly/{month_dir}_{week_name}.md")

    # Monthly summaries
    for month_dir in sorted(CHRONICLE_DIR.iterdir()):
        if not month_dir.is_dir():
            continue
        if not list(month_dir.rglob("*.md")):
            continue  # empty month directory — nothing to summarise
        if not month_has_ended(month_dir.name):
            print(f"Skipping monthly/{month_dir.name}.md — month not yet ended")
            continue
        written = summarise_month(month_dir.name, skip_existing=True)
        if written:
            months_done += 1
        else:
            skipped.append(f"monthly/{month_dir.name}.md")

    print(f"\nAll done. {weeks_done} weekly and {months_done} monthly summaries written.")
    if skipped:
        print(f"{len(skipped)} already existed and were skipped.", end="")
        if list_skipped:
            print()
            for s in skipped:
                print(f"  skipped: summaries/{s}")
        else:
            print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0]

    if command == "week":
        if len(args) != 3:
            sys.exit(
                "Usage: summarise.py week <month_dir> <week_dir>\n"
                "Example: summarise.py week nov_2025 week_2"
            )
        summarise_week(args[1], args[2])

    elif command == "month":
        if len(args) != 2:
            sys.exit(
                "Usage: summarise.py month <month_dir>\n"
                "Example: summarise.py month nov_2025"
            )
        summarise_month(args[1])

    elif command == "all":
        extra = args[1:]
        if extra and extra != ["--list-skipped"]:
            sys.exit("Usage: summarise.py all [--list-skipped]")
        summarise_all(list_skipped="--list-skipped" in extra)

    else:
        sys.exit(f"Unknown command '{command}'. Use 'week', 'month', or 'all'.")


if __name__ == "__main__":
    main()
