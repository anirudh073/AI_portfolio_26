"""
Researcher — AI-driven research workflow using Gemini agent pair.

Workflow:
  1. Analysis agent reads AnAge data, proposes a thesis project
  2. Analysis agent performs analysis and writes results
  3. Critique agent reviews results (configurable rounds)
  4. Analysis agent revises based on critique
  5. All outputs saved to output/ directory

Usage:
  python researcher.py [--rounds N] [--model MODEL]
"""

import os
import sys
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not found in .env")

DATA_FILE = Path(__file__).parent / "data" / "anage_data.txt"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_MODEL = "gemini-2.5-pro"
CRITIQUE_MODEL = "gemini-2.5-pro"

# Token budget warning threshold (approximate, based on Gemini free/paid limits)
TOKEN_WARN_THRESHOLD = 500_000

# ── System prompts ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a rigorous research scientist specialising in comparative biology and life-history theory.
You are given raw biological data and tasked with:
1. Proposing a well-defined research question suitable for a bachelor's thesis or higher
2. Designing and executing an appropriate statistical analysis
3. Writing up results to a high academic standard

Be specific, quantitative, and honest about limitations. Do not hallucinate statistics — work only from the data provided.
When writing results, use clear section headers: Background, Research Question, Methods, Results, Discussion, Limitations."""

CRITIQUE_SYSTEM = """You are a senior academic reviewer at a biology journal. Your job is to critically evaluate research outputs.

For each review, assess:
- Scientific validity of the research question
- Appropriateness of the statistical methods chosen
- Quality and accuracy of the results
- Strength of the interpretation and discussion
- Missing analyses that would strengthen the paper
- Any potential errors, overstatements, or hallucinations

Be direct and constructive. Number your critiques. End with an overall verdict: ACCEPT / MINOR REVISION / MAJOR REVISION."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data_summary() -> str:
    """Load AnAge data and return a structured summary for the model."""
    import pandas as pd

    df = pd.read_csv(DATA_FILE, sep="\t")

    lines = [
        f"Dataset: AnAge — Animal Ageing and Longevity Database (Build 15, 2023)",
        f"Shape: {df.shape[0]} species × {df.shape[1]} columns",
        "",
        "=== COLUMNS ===",
        ", ".join(df.columns.tolist()),
        "",
        "=== CLASS DISTRIBUTION ===",
        df["Class"].value_counts().to_string(),
        "",
        "=== MISSING DATA (%) ===",
        (df.isnull().mean() * 100).round(1).to_string(),
        "",
        "=== LONGEVITY SUMMARY ===",
        df["Maximum longevity (yrs)"].describe().to_string(),
        "",
        "=== TOP 10 LONGEST-LIVED ===",
        df[["Common name", "Class", "Maximum longevity (yrs)"]]
            .dropna()
            .sort_values("Maximum longevity (yrs)", ascending=False)
            .head(10)
            .to_string(index=False),
        "",
        "=== SAMPLE ROWS (first 5) ===",
        df.head(5).to_string(),
        "",
        "=== FULL DATA (tab-separated) ===",
        df.to_csv(sep="\t", index=False),
    ]
    return "\n".join(lines)


def save_output(filename: str, content: str):
    """Save output to file with timestamp header."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        f.write(f"Generated: {ts}\n{'='*60}\n\n{content}")
    print(f"  → Saved: {path}")
    return path


def print_section(title: str, content: str, width: int = 80):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    print(content[:3000] + ("\n[...truncated for display...]" if len(content) > 3000 else ""))


def warn_tokens(total: int):
    if total > TOKEN_WARN_THRESHOLD:
        print(f"\n  ⚠ WARNING: ~{total:,} tokens used this run. "
              f"Monitor your Gemini quota at https://aistudio.google.com/")


# ── Agent calls ───────────────────────────────────────────────────────────────

def call_agent(client, model: str, system: str, messages: list[dict], label: str) -> tuple[str, int]:
    """Call a Gemini model with a message history. Returns (response_text, token_count)."""
    print(f"\n  [{label}] calling {model}...")

    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
        ),
    )

    text = response.text
    tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
    print(f"  [{label}] done (~{tokens:,} tokens)")
    return text, tokens


# ── Main workflow ──────────────────────────────────────────────────────────────

def run(critique_rounds: int = 2):
    client = genai.Client(api_key=API_KEY)
    total_tokens = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("  RESEARCHER — AI-driven research workflow")
    print(f"  Analysis: {ANALYSIS_MODEL}  |  Critique: {CRITIQUE_MODEL}")
    print(f"  Critique rounds: {critique_rounds}")
    print("="*60)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading AnAge data...")
    data_summary = load_data_summary()
    print(f"  Data loaded ({len(data_summary):,} chars)")

    # ── Step 2: Thesis proposal + analysis ────────────────────────────────────
    print("\n[2/4] Analysis agent: propose thesis + perform analysis...")

    analysis_messages = [
        {
            "role": "user",
            "content": textwrap.dedent(f"""
                Here is the AnAge dataset — raw biological data on animal ageing and life history.

                {data_summary}

                Your task:
                1. Propose a specific, testable research question suitable for a bachelor's thesis or higher.
                   The question should be interesting to a broad biology audience (evolutionary biology, ecology, physiology).
                2. Design and carry out an appropriate statistical analysis of this data to address your question.
                3. Write a full research results document with these sections:
                   - Background
                   - Research Question & Hypothesis
                   - Methods
                   - Results (include specific statistics, effect sizes, p-values where relevant)
                   - Discussion
                   - Limitations

                Be rigorous. Work only from the data provided. Include specific numbers throughout.
            """).strip()
        }
    ]

    analysis_result, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, "ANALYSIS"
    )
    total_tokens += tokens

    save_output(f"{timestamp}_01_analysis.md", analysis_result)
    print_section("ANALYSIS RESULT (preview)", analysis_result)

    # Add to message history
    analysis_messages.append({"role": "model", "content": analysis_result})

    # ── Step 3: Critique rounds ────────────────────────────────────────────────
    print(f"\n[3/4] Running {critique_rounds} critique round(s)...")

    for round_num in range(1, critique_rounds + 1):
        print(f"\n  --- Critique Round {round_num}/{critique_rounds} ---")

        # Critique agent reviews the latest analysis
        critique_messages = [
            {
                "role": "user",
                "content": textwrap.dedent(f"""
                    Please critically review the following research document (Round {round_num}).
                    The data source is the AnAge Animal Ageing and Longevity Database (4,645 species).

                    === RESEARCH DOCUMENT ===
                    {analysis_result}
                    ========================

                    Provide a numbered critique. Be specific — cite exact claims that need support or correction.
                    End with: VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]
                """).strip()
            }
        ]

        critique_result, tokens = call_agent(
            client, CRITIQUE_MODEL, CRITIQUE_SYSTEM, critique_messages, f"CRITIQUE R{round_num}"
        )
        total_tokens += tokens

        save_output(f"{timestamp}_02_critique_r{round_num}.md", critique_result)
        print_section(f"CRITIQUE ROUND {round_num} (preview)", critique_result)

        warn_tokens(total_tokens)

        # Analysis agent revises based on critique
        analysis_messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""
                A senior reviewer has critiqued your research document (Round {round_num} of {critique_rounds}):

                === CRITIQUE ===
                {critique_result}
                ================

                Please revise your research document addressing the reviewer's points.
                Maintain all sections. Be specific about what you changed and why.
                If you disagree with any critique, state your reasoning.
            """).strip()
        })

        analysis_result, tokens = call_agent(
            client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, f"REVISION R{round_num}"
        )
        total_tokens += tokens

        analysis_messages.append({"role": "model", "content": analysis_result})
        save_output(f"{timestamp}_03_revision_r{round_num}.md", analysis_result)
        print_section(f"REVISION ROUND {round_num} (preview)", analysis_result)

        warn_tokens(total_tokens)

    # ── Step 4: Final outputs ──────────────────────────────────────────────────
    print("\n[4/4] Generating final outputs...")

    # Final clean results document
    analysis_messages.append({
        "role": "user",
        "content": textwrap.dedent("""
            Now produce the final, polished research document incorporating all revisions.
            This is the version that will be presented at a biology institute conference.
            Format it cleanly with clear section headers. It should stand alone without needing context.
        """).strip()
    })

    final_result, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, "FINAL DOCUMENT"
    )
    total_tokens += tokens
    save_output(f"{timestamp}_04_final_results.md", final_result)

    # Poster brief
    poster_messages = [
        {
            "role": "user",
            "content": textwrap.dedent(f"""
                Convert the following research document into a conference poster layout.
                The poster will be presented at a biology institute to a broad scientific audience.

                Use this structure:
                - TITLE (punchy, max 15 words)
                - AUTHORS & AFFILIATION (placeholder: "AI Research Demo, NCBS Bangalore, 2026")
                - INTRODUCTION (3-4 bullet points, key background)
                - RESEARCH QUESTION (1 sentence)
                - METHODS (brief, 3-5 bullet points)
                - KEY RESULTS (use bullet points, include key numbers/stats)
                - FIGURES (describe 2-3 figures that should appear on the poster — axes, what they show)
                - CONCLUSIONS (3-4 bullet points)
                - ACKNOWLEDGEMENTS

                === RESEARCH DOCUMENT ===
                {final_result}
            """).strip()
        }
    ]

    poster_result, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, poster_messages, "POSTER"
    )
    total_tokens += tokens
    save_output(f"{timestamp}_05_poster_layout.md", poster_result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WORKFLOW COMPLETE")
    print(f"  Total tokens used: ~{total_tokens:,}")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.glob(f"{timestamp}_*.md")):
        print(f"    {f.name}")
    print("="*60)

    warn_tokens(total_tokens)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-driven research workflow on AnAge dataset")
    parser.add_argument("--rounds", type=int, default=2, help="Number of critique rounds (default: 2)")
    args = parser.parse_args()

    run(critique_rounds=args.rounds)
