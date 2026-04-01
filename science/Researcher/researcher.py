"""
Researcher — AI-driven research workflow using Gemini agent pair.

Workflow:
  1. Analysis agent reads AnAge data, proposes and executes Analysis 1 (broad opening question)
  2. Analysis agent proposes and executes Analysis 2, motivated by findings from Analysis 1
  3. (Optional) Analysis 3, motivated by Analysis 2
  4. Critique agent reviews the full body of work as a coherent story
  5. Analysis agent revises (configurable rounds)
  6. Final synthesis document + conference poster with story arc and future directions

Usage:
  python researcher.py [--analyses N] [--rounds N]
  python researcher.py --analyses 2 --rounds 2   (default)
  python researcher.py --analyses 3 --rounds 1
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

ANALYSIS_MODEL = "gemini-3.1-pro-preview"
CRITIQUE_MODEL = "gemini-2.5-pro"

TOKEN_WARN_THRESHOLD = 5_000_000  # paid account

# ── System prompts ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a rigorous research scientist specialising in comparative biology and life-history theory.
Your goal is to produce a coherent body of work — a sequence of analyses that build on each other and tell a compelling scientific story.

Rules:
- Be specific and quantitative. Include effect sizes, p-values, confidence intervals.
- Do not hallucinate statistics — work only from the data provided.
- Each analysis must be clearly motivated by the previous one.
- Acknowledge limitations honestly.
- Write to the standard of a published paper, not a student report.
- Write like a human scientist, not an AI. Avoid AI slop: no "certainly!", no "it is important to note that",
  no "in conclusion, this study has shown", no "fascinating", no "crucially", no "it is worth noting",
  no hollow throat-clearing phrases. Be direct, specific, and dry where appropriate. Let the results speak."""

CRITIQUE_SYSTEM = """You are a senior academic reviewer at a biology journal evaluating a multi-part research project.
Your job is to assess the work as a whole — does it tell a coherent scientific story?

For each review, assess:
1. Scientific validity of each research question
2. Whether each analysis is genuinely motivated by the previous one (narrative coherence)
3. Appropriateness of statistical methods
4. Quality and accuracy of results
5. Strength of interpretation and discussion
6. What is missing — analyses, controls, or comparisons that would strengthen the story
7. Any errors, overstatements, or hallucinations
8. Any AI slop language — flag hollow phrases, over-hedging, or unnatural academic prose

Be direct and constructive. Number your critiques. End with:
VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data_summary() -> str:
    import pandas as pd
    df = pd.read_csv(DATA_FILE, sep="\t")
    lines = [
        "Dataset: AnAge — Animal Ageing and Longevity Database (Build 15, 2023)",
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
    print(content[:2000] + ("\n[...truncated for display...]" if len(content) > 2000 else ""))


def warn_tokens(total: int):
    if total > TOKEN_WARN_THRESHOLD:
        pct = total / 1_500_000 * 100  # Gemini free tier ~1.5M tokens/day
        print(f"\n  ⚠  TOKEN WARNING: ~{total:,} tokens used (~{pct:.0f}% of free daily limit).")
        print(f"     Monitor at: https://aistudio.google.com/")


def call_agent(client, model: str, system: str, messages: list[dict], label: str) -> tuple[str, int]:
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

def run(n_analyses: int = 2, critique_rounds: int = 2):
    client = genai.Client(api_key=API_KEY)
    total_tokens = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyses = []  # stores (title, result) for each analysis

    print("\n" + "="*60)
    print("  RESEARCHER — AI-driven research workflow")
    print(f"  Model: {ANALYSIS_MODEL}")
    print(f"  Analyses: {n_analyses}  |  Critique rounds: {critique_rounds}")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[SETUP] Loading AnAge data...")
    data_summary = load_data_summary()
    print(f"  Data loaded ({len(data_summary):,} chars)")

    # ── Analysis chain ─────────────────────────────────────────────────────────
    analysis_messages = []

    for i in range(1, n_analyses + 1):
        step_label = f"ANALYSIS {i}/{n_analyses}"
        print(f"\n{'─'*60}")
        print(f"  {step_label}")
        print(f"{'─'*60}")

        if i == 1:
            prompt = textwrap.dedent(f"""
                Here is the AnAge dataset — raw biological data on animal ageing and life history.

                {data_summary}

                TASK — Analysis 1 of {n_analyses}:
                You are beginning a multi-part research project. This first analysis establishes
                the central question and findings. Subsequent analyses will build on it.

                1. Propose a specific, testable research question suitable for a bachelor's thesis or higher.
                   Choose something interesting to a broad biology audience at a research institute.
                   Think about what question will naturally lead to a deeper follow-up.
                2. Design and execute an appropriate statistical analysis.
                3. Write up the results in full:
                   - Background
                   - Research Question & Hypothesis
                   - Methods
                   - Results (with statistics: effect sizes, p-values, CIs)
                   - Discussion
                   - Open Questions (what does this finding raise that a follow-up analysis could address?)
                   - Limitations

                Be rigorous. Work only from the data provided.
            """).strip()

        else:
            prior_summaries = "\n\n".join([
                f"=== ANALYSIS {j}: {title} ===\n{result}"
                for j, (title, result) in enumerate(analyses, 1)
            ])
            prompt = textwrap.dedent(f"""
                You are continuing a multi-part research project on the AnAge dataset.
                Here is the full dataset again for reference:

                {data_summary}

                === PRIOR ANALYSES ===
                {prior_summaries}
                ======================

                TASK — Analysis {i} of {n_analyses}:
                Based on the open questions and findings from the previous analysis,
                propose and execute the next logical analysis. It must:
                - Be clearly motivated by a specific finding or open question from the prior work
                - Deepen or extend the story (not repeat it)
                - Use a different angle, subset, or method where appropriate

                Write up in full:
                - Motivation (what finding from prior analysis inspired this?)
                - Research Question & Hypothesis
                - Methods
                - Results (with statistics)
                - Discussion
                - Open Questions (for the next analysis, or future work)
                - Limitations
            """).strip()

        analysis_messages.append({"role": "user", "content": prompt})

        result, tokens = call_agent(
            client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, step_label
        )
        total_tokens += tokens
        analysis_messages.append({"role": "model", "content": result})

        # Extract a short title from the result for tracking
        title_line = next((l.strip("#* ") for l in result.splitlines() if l.strip()), f"Analysis {i}")
        title = title_line[:80]
        analyses.append((title, result))

        save_output(f"{timestamp}_analysis_{i:02d}.md", result)
        print_section(f"{step_label} RESULT (preview)", result)
        warn_tokens(total_tokens)

    # ── Critique rounds ────────────────────────────────────────────────────────
    if critique_rounds > 0:
        print(f"\n{'─'*60}")
        print(f"  CRITIQUE PHASE ({critique_rounds} round(s))")
        print(f"{'─'*60}")

    current_body = "\n\n".join([
        f"=== ANALYSIS {j}: {title} ===\n{result}"
        for j, (title, result) in enumerate(analyses, 1)
    ])

    for round_num in range(1, critique_rounds + 1):
        print(f"\n  --- Critique Round {round_num}/{critique_rounds} ---")

        critique_messages = [{
            "role": "user",
            "content": textwrap.dedent(f"""
                Please critically review this multi-part research project (Round {round_num}).
                The data source is the AnAge Animal Ageing and Longevity Database (4,645 species).
                Evaluate it as a coherent body of work — does it tell a compelling scientific story?

                {current_body}

                Provide a numbered critique. Be specific. Cite exact claims that need support or correction.
                Pay particular attention to: narrative coherence between analyses, statistical validity,
                and what would make this poster-worthy for a biology institute audience.
                End with: VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]
            """).strip()
        }]

        critique_result, tokens = call_agent(
            client, CRITIQUE_MODEL, CRITIQUE_SYSTEM, critique_messages, f"CRITIQUE R{round_num}"
        )
        total_tokens += tokens
        save_output(f"{timestamp}_critique_r{round_num}.md", critique_result)
        print_section(f"CRITIQUE ROUND {round_num} (preview)", critique_result)
        warn_tokens(total_tokens)

        # Analysis agent revises the full body
        analysis_messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""
                A senior reviewer has critiqued the full body of work (Round {round_num} of {critique_rounds}):

                === CRITIQUE ===
                {critique_result}
                ================

                Please revise ALL analyses to address the reviewer's points.
                Present them in sequence. Make the narrative arc stronger.
                State explicitly what you changed in each analysis and why.
                If you disagree with any point, explain your reasoning.
            """).strip()
        })

        revised_body, tokens = call_agent(
            client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, f"REVISION R{round_num}"
        )
        total_tokens += tokens
        analysis_messages.append({"role": "model", "content": revised_body})
        current_body = revised_body
        save_output(f"{timestamp}_revision_r{round_num}.md", revised_body)
        print_section(f"REVISION ROUND {round_num} (preview)", revised_body)
        warn_tokens(total_tokens)

    # ── Final synthesis document ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  FINAL SYNTHESIS")
    print(f"{'─'*60}")

    analysis_messages.append({
        "role": "user",
        "content": textwrap.dedent(f"""
            Now produce the final, polished research document — a synthesis of all {n_analyses} analyses.
            This will be presented at a biology institute conference.

            Structure it as a single cohesive paper:
            - Abstract (150 words max)
            - Introduction & Background
            - Analysis 1: [title] — Methods, Results, Discussion
            - Analysis 2: [title] — Motivation, Methods, Results, Discussion
            {"- Analysis 3: [title] — Motivation, Methods, Results, Discussion" if n_analyses >= 3 else ""}
            - Overall Discussion (synthesise the story across all analyses)
            - Conclusions (3-5 key take-home messages)
            - Future Directions (3-5 concrete next experiments or questions)
            - Limitations

            The document should stand alone. Include all key statistics.
            Write for a broad biology audience — assume no prior context.
        """).strip()
    })

    final_result, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, "FINAL SYNTHESIS"
    )
    total_tokens += tokens
    save_output(f"{timestamp}_final_synthesis.md", final_result)
    print_section("FINAL SYNTHESIS (preview)", final_result)

    # ── Plain-English writeup ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PLAIN-ENGLISH WRITEUP")
    print(f"{'─'*60}")

    writeup_messages = [{
        "role": "user",
        "content": textwrap.dedent(f"""
            Write a brief, accessible summary of the following research for a broad biology audience.
            Assume they are scientists but not specialists in life-history theory.

            Requirements:
            - 400-600 words
            - No jargon — if a technical term is needed, explain it in plain English
            - Tell it as a story: what question was asked, what was found, why it matters
            - Highlight the most surprising or counterintuitive finding
            - End with 2-3 sentences on what this opens up for future research
            - Tone: engaging, like a Nature News & Views piece, not a textbook

            === RESEARCH SYNTHESIS ===
            {final_result}
        """).strip()
    }]

    writeup_result, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, writeup_messages, "WRITEUP"
    )
    total_tokens += tokens
    save_output(f"{timestamp}_writeup.md", writeup_result)
    print_section("PLAIN-ENGLISH WRITEUP (preview)", writeup_result)

    # ── Conference poster (HTML) ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  CONFERENCE POSTER (HTML)")
    print(f"{'─'*60}")

    poster_messages = [{
        "role": "user",
        "content": textwrap.dedent(f"""
            Convert the following research into a self-contained HTML conference poster.

            DESIGN REQUIREMENTS:
            - A0 portrait format (841mm × 1189mm), print-ready
            - Clean academic style: white background, two-column layout below the header
            - Header: full-width banner with title (large, bold), authors, affiliation
            - Colour scheme: deep navy (#1a2e4a) for headers, white text in header, black body text,
              light grey (#f5f5f5) panel backgrounds, accent colour (#2e7d5e) for highlights
            - Sections as distinct panels with subtle borders
            - Font: system sans-serif stack, title ~48px, section headers ~24px, body ~16px
            - All CSS inline or in a <style> block — single file, no external dependencies

            POSTER DESIGN RULES (follow strictly):
            - Must be readable from 3-4 feet away: body text minimum 20px, figure captions 18px,
              section headers 28px+, title 52px+
            - Minimal text: use bullet points, not paragraphs. If a sentence can be cut, cut it.
              The figures carry the story — text supports them, not the other way around.
            - Every section heading must be DESCRIPTIVE, not generic.
              BAD: "Results", "Introduction", "Conclusions"
              GOOD: "Larger animals live longer — but not equally so", "Developmental pace, not
              body temperature, drives longevity", "What this means for ageing research"
            - Figures must be poster-optimised: large, simple, high contrast. Each figure placeholder
              must specify: title (descriptive), x-axis, y-axis, key data shown, colour scheme,
              and the single take-home message a viewer should get in 5 seconds.
            - No dense tables. Use large-type callout numbers (e.g. "R² = 0.80") styled prominently
              instead of tabular results where possible.

            CONTENT STRUCTURE:
            - HEADER PANEL (full width): Title · Authors · Affiliation · QR placeholder (grey box, right side)
            - LEFT COLUMN:
                • Why this matters (3 bullets max — each a complete thought, no filler)
                • The Story: Act I / Act II (2 short punchy paragraphs, one per analysis)
                • 2-3 large callout statistics (big number + one-line explanation)
            - RIGHT COLUMN:
                • Figures ({n_analyses + 1} figure placeholders — large grey boxes, each with a
                  descriptive title, full axis/data spec, and 1-sentence take-home caption)
                • What we conclude (4-5 bullets — each a finding, stated as a claim, not a hedge)
                • Where this leads (3 bullets — specific future directions, not vague gestures)
            - FOOTER (full width): Acknowledgements · Data source · small print

            Output ONLY valid HTML. No markdown, no explanation, no code fences. Start with <!DOCTYPE html>.

            === RESEARCH SYNTHESIS ===
            {final_result}

            === PLAIN-ENGLISH WRITEUP ===
            {writeup_result}
        """).strip()
    }]

    poster_html, tokens = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, poster_messages, "POSTER HTML"
    )
    total_tokens += tokens

    # Strip any accidental markdown code fences
    poster_html = poster_html.strip()
    if poster_html.startswith("```"):
        poster_html = "\n".join(poster_html.splitlines()[1:])
    if poster_html.endswith("```"):
        poster_html = "\n".join(poster_html.splitlines()[:-1])

    poster_path = OUTPUT_DIR / f"{timestamp}_poster.html"
    with open(poster_path, "w") as f:
        f.write(poster_html)
    print(f"  → Saved: {poster_path}")
    print(f"  → Open in browser, then File → Print → Save as PDF (A0, no margins)")

    # Also save text layout for reference
    save_output(f"{timestamp}_poster_layout.md", poster_html[:500] + "\n\n[Full HTML saved as .html]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WORKFLOW COMPLETE")
    print(f"  Analyses: {n_analyses}  |  Critique rounds: {critique_rounds}")
    print(f"  Total tokens: ~{total_tokens:,}")
    print(f"  Outputs:")
    for f in sorted(OUTPUT_DIR.glob(f"{timestamp}_*.md")):
        print(f"    {f.name}")
    print("="*60)
    warn_tokens(total_tokens)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-driven multi-analysis research workflow on AnAge")
    parser.add_argument("--analyses", type=int, default=2, choices=[1, 2, 3],
                        help="Number of chained analyses (default: 2, max: 3)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of critique rounds (default: 2)")
    args = parser.parse_args()
    run(n_analyses=args.analyses, critique_rounds=args.rounds)
