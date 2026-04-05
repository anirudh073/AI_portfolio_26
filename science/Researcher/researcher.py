"""
Researcher — AI-driven research workflow using Gemini agent pair.

Workflow:
  1. Analysis agent reads the dataset, proposes and executes Analysis 1 (broad opening question)
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
OUTPUT_BASE = Path(__file__).parent / "output"
OUTPUT_BASE.mkdir(exist_ok=True)
OUTPUT_DIR = OUTPUT_BASE  # overridden per-run in run() and --poster-only

ANALYSIS_MODEL = "gemini-3.1-pro-preview"
CRITIQUE_MODEL = "gemini-3.1-pro-preview"

TOKEN_WARN_THRESHOLD = 5_000_000  # paid account

# ── System prompts ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a rigorous research scientist. Your goal is to produce a coherent body of work —
a sequence of analyses that build on each other and tell a compelling scientific story.

Rules:
- Be specific and quantitative. Include effect sizes, p-values, confidence intervals.
- Do not hallucinate statistics — work only from the data provided.
- Each analysis must be clearly motivated by the previous one.
- Acknowledge limitations honestly.
- Write to the standard of a published paper, not a student report.
- Write like a human scientist, not an AI. Avoid AI slop: no "certainly!", no "it is important to note that",
  no "in conclusion, this study has shown", no "fascinating", no "crucially", no "it is worth noting",
  no hollow throat-clearing phrases. Be direct, specific, and dry where appropriate. Let the results speak."""

CRITIQUE_SYSTEM = """You are a senior academic reviewer at a peer-reviewed scientific journal. You are direct, exacting, and do not waste words.
Do not open with pleasantries — begin immediately with your assessment.

When reviewing analyses, assess:
1. Scientific validity of each research question
2. Whether each analysis is genuinely motivated by the previous one (narrative coherence)
3. Appropriateness of statistical methods chosen
4. Accuracy and precision of results — flag any overstatements or understatements
5. Strength of interpretation and discussion
6. Missing analyses, controls, or comparisons that would strengthen the story
7. Errors, overstatements, or hallucinations
8. AI slop language — hollow phrases, over-hedging, unnatural academic prose

When reviewing a poster, additionally assess:
1. Narrative flow — can a first-time viewer follow the story in under 5 minutes?
2. Is the central question clearly stated before the answer is given?
3. Do headings state findings/claims, or are they generic labels?
4. Is Act I tension preserved (finding + open question), or is Act II's answer leaked into Act I?
5. Do the figures support the narrative in order, or are they decorative?
6. Is there any text that could be cut without losing meaning?
7. Would a scientist standing 3 feet away understand what was found?

Be direct and constructive. Number every critique. Be specific — quote exact text that needs changing.
End with: VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]"""

POSTER_CRITIQUE_SYSTEM = """You are a science communication expert and conference poster judge with 20 years of experience
evaluating research posters at international scientific conferences.
You care above all about clarity, narrative, and visual impact. You do not start with pleasantries.

Evaluate this poster on:
1. NARRATIVE FLOW: Does it guide the viewer from question → finding → implication in a logical sequence?
2. QUESTION CLARITY: Is the central research question explicitly stated before the answer is given?
3. HEADINGS: Are all headings descriptive claims/findings, or generic labels like "Results", "Introduction"?
4. QUESTION-ANSWER LOGIC: Does each panel pose a question and answer it, leading naturally to the next? Is there any panel that answers before establishing the question?
5. TEXT ECONOMY: Is every sentence earning its space? Flag anything that can be cut.
6. FIGURES: Do the figures tell the story in sequence? Are they self-explanatory with captions alone?
7. READABILITY: Would someone standing 3-4 feet away understand the take-home message in 60 seconds?
8. SCIENTIFIC ACCURACY: Are the claims on the poster consistent with the underlying research?

Quote specific text when you flag issues. Number every critique.
End with: VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data_summary() -> str:
    import pandas as pd

    # Auto-detect delimiter
    suffix = DATA_FILE.suffix.lower()
    sep = "\t" if suffix in (".txt", ".tsv") else ","
    df = pd.read_csv(DATA_FILE, sep=sep)

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    lines = [
        f"Dataset file: {DATA_FILE.name}",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "",
        "=== COLUMNS ===",
        ", ".join(df.columns.tolist()),
        "",
        "=== MISSING DATA (%) ===",
        (df.isnull().mean() * 100).round(1).to_string(),
        "",
    ]

    if categorical_cols:
        lines += [
            "=== CATEGORICAL COLUMN DISTRIBUTIONS (top 10 values each) ===",
        ]
        for col in categorical_cols[:5]:  # limit to first 5 to avoid bloat
            lines.append(f"-- {col} --")
            lines.append(df[col].value_counts().head(10).to_string())
            lines.append("")

    if numeric_cols:
        lines += [
            "=== NUMERIC COLUMNS SUMMARY ===",
            df[numeric_cols].describe().to_string(),
            "",
        ]

    lines += [
        "=== SAMPLE ROWS (first 5) ===",
        df.head(5).to_string(),
        "",
        "=== FULL DATA ===",
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


def call_agent(client, model: str, system: str, messages: list[dict], label: str,
               max_retries: int = 4) -> tuple[str, int, float]:
    import time
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        print(f"\n  [{label}] calling {model} (attempt {attempt}/{max_retries})...")
        try:
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
            elapsed = time.time() - t0
            print(f"  [{label}] done  ~{tokens:,} tokens  {elapsed:.0f}s")
            return text, tokens, elapsed
        except Exception as e:
            err_str = str(e)
            if "400" in err_str and "INVALID_ARGUMENT" in err_str:
                print(f"  [{label}] FATAL: context too long — not retrying")
                raise
            if attempt == max_retries:
                raise
            wait = 15 * attempt  # 15s, 30s, 45s
            print(f"  [{label}] error: {e}")
            print(f"  [{label}] retrying in {wait}s...")
            time.sleep(wait)


# ── Poster generation (standalone) ───────────────────────────────────────────

def sanitize_poster_html(html: str) -> str:
    """Fix common model tag errors: mismatched </script>/<style> closing tags."""
    import re
    # Pass 1: </script> wrongly closing a <style> block → replace with </style>
    html = re.sub(
        r'(<style\b[^>]*>)(.*?)(</script>)',
        lambda m: m.group(1) + m.group(2) + '</style>',
        html, flags=re.DOTALL
    )
    # Pass 2: </style> wrongly closing an inline <script> block → replace with </script>
    # Only match inline scripts (no src attribute)
    html = re.sub(
        r'(<script(?:\s(?!src)[^>]*)?>)(.*?)(</style>)',
        lambda m: m.group(1) + m.group(2) + '</script>',
        html, flags=re.DOTALL
    )
    return html


def strip_html_for_critique(html: str) -> str:
    """Extract visible text + structure from HTML for critique (strips JS/CSS to save tokens)."""
    import re
    # Remove <style> and <script> blocks entirely
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_poster(client, final_result: str, writeup_result: str, n_analyses: int,
                    timestamp: str, with_critique: bool = True) -> Path:
    """Generate the HTML conference poster. Can be called standalone for reruns."""
    print(f"\n{'─'*60}")
    print("  CONFERENCE POSTER (HTML)")
    print(f"{'─'*60}")

    poster_messages = [{
        "role": "user",
        "content": textwrap.dedent(f"""
            Convert the following research into a self-contained HTML conference poster.
            You MUST render all figures as actual charts using Chart.js
            (load from https://cdn.jsdelivr.net/npm/chart.js).
            Do NOT use placeholder boxes or text descriptions for figures — every figure
            must be a real <canvas> element with Chart.js code and actual data values
            from the research embedded as JavaScript arrays.

            STRICT DATA RULE: All chart data values must be taken verbatim from the synthesis
            document. Do not estimate, round, or approximate any value. Do not invent data points.
            If a specific species value is not explicitly stated in the synthesis, do not plot it.

            STRICT HTML RULE: No LaTeX notation anywhere. No $...$ delimiters, no \\(...\\).
            Write p=0.004, use R² (Unicode), <sup> tags for superscripts. Plain HTML only.

            STRICT TEXT RULE: Use bullet points and short punchy sentences — no dense prose.
            Narrative panels: max 3-4 sentences each, under 30 words per sentence.

            DESIGN REQUIREMENTS:
            - LANDSCAPE format: 4 feet wide × 3 feet tall (121.9cm × 91.4cm). Set .poster to
              width: 1219mm; height: 914mm in CSS.
            - Viewport scaling wrapper: wrap .poster in a div.viewport-scaler with
              transform: scale(0.45); transform-origin: top left;
              Add a banner above: "Zoom out to see full poster / Print at 4ft×3ft landscape, no margins"
            - NO WHITESPACE: The poster must fill the entire 4ft×3ft area with zero empty space.
              Use CSS flexbox or grid with flex: 1 / flex-grow so panels and figures expand to
              fill all available height. No large margins or padding. Column gap max 20px.
              Header and footer should be compact (no excessive padding). Every pixel must be used.
            - Clean academic style: white background, header full-width, then a MULTI-COLUMN layout.
              Choose the number of columns (2–4) to best fit the content — more analyses = more columns.
              Keep all columns equal width. Never use more than 4 columns.
            - Header: full-width banner with title (large, bold), authors, affiliation
            - Colour scheme: deep navy (#1a2e4a) headers, white header text, black body text,
              light grey (#f5f5f5) panel backgrounds, accent (#2e7d5e) highlights
            - Font: system sans-serif, title 90px+, section headers 40px+, body 28px+, captions 24px+
            - All CSS in a <style> block; JS in <script> blocks — single file, no other dependencies

            POSTER DESIGN RULES:
            - Readable from 3-4 feet: body 28px min, captions 24px min, headers 40px+, title 90px+
            - Every heading DESCRIPTIVE not generic.
              BAD: "Results" GOOD: "Great apes outlive lesser apes — body mass explains half of it"
            - No dense tables. Use large-type stat callout cards instead.

            FIGURE DESIGN RULES (apply to all figures, regardless of research topic):
            - ALL chart canvas wrappers must use a SINGLE shared CSS class (e.g. .chart-wrapper)
              with an IDENTICAL fixed height (e.g. height: 420px). Never set height inline or
              per-figure. Never use min-height or max-height on figures — this causes one figure
              to balloon and fill remaining space. Every chart must be exactly the same height.
            - All figure containers must use flex: 1 so they share column space equally.
              No figure should ever be taller than any other.
            - Prefer DIRECT LABELS on data points over legends. If the dataset is small enough
              to label individually (n < 30), label each point/bar directly with its name.
              If a legend is unavoidable, place it below the chart — never overlapping data — font 22px+.
            - Use a CONSISTENT colour scheme across ALL figures. Assign one colour per group or
              category and never deviate. The viewer should instantly know which colour means which
              group without re-reading the legend each time.
            - ALL figures must use a CONSISTENT dataset — every data point that appears in one
              figure should be traceable across the others. Never silently drop data points between figures.
            - Do not add derived summary markers (means, medians) on top of individual data points
              unless the summary IS the point of the figure. Mixing raw and summary data in the
              same figure adds visual clutter.
            - Axis labels: 24px bold. Tick labels: 20px. Figure title: 28px bold above canvas.
              Point radius: minimum 10px. Bar height: minimum 30px. Charts must be large enough
              that all labels are readable without zooming.
            - Each figure must have a single, self-evident take-home message. If a viewer cannot
              state what the figure shows in one sentence, it is too complex — simplify it.

            FIGURES ({n_analyses + 1} total, right column, derived from the synthesis):
            Choose the most informative chart types for this specific research. The figures should
            follow the same logical order as the narrative — each figure supports the corresponding
            step in the story. Use only data values stated explicitly in the synthesis document.
            Each figure needs: a descriptive title (the finding, not the variable), Chart.js canvas,
            and a one-sentence caption below stating the take-home message.

            NARRATIVE FLOW — the poster must follow a logical sequence that a viewer can read
            left-to-right, top-to-bottom without prior context. Each panel should pose a question
            or make an observation, which is then answered or supported — leading naturally to the
            next question. The final panel should leave the viewer with a clear take-home message
            and a sense of what comes next.

            The flow should follow the logic of the research itself:
            question → answer → this raises a new question → answer → implication.
            Do not jump to conclusions before establishing the evidence. Do not repeat findings.
            Each panel must earn its place by advancing the story.

            HEADING RULES — every heading must be a descriptive claim or finding, never a label.
            BAD: "Why this matters" → GOOD: "Human aging is an unsolved evolutionary puzzle"
            BAD: "Results" → GOOD: "Great apes outlive lesser apes — but body mass explains half of it"
            BAD: "What we conclude" → GOOD: "Great apes age by the rules — humans don't"
            BAD: "Where this leads" → GOOD: "Three open questions this finding raises"
            BAD: "Key findings" → GOOD: "Three numbers that tell the story"

            CONTENT STRUCTURE:
            - HEADER (full width): Title (the conclusion, stated boldly) ·
              "AI Research Demo, NCBS Bangalore, 2026" · QR placeholder right
            - COLUMNS (2–4, choose based on content volume):
                • Column 1 always starts with: opening panel (central question, 2-3 sentences),
                  then narrative panels walking the story (one question-answer per panel).
                • Spread figures, stat cards, conclusions and future directions across the remaining
                  columns in narrative order — figures go next to the panel that describes them.
                • 3 stat callout cards (big number + one-line explanation, no LaTeX) distributed
                  where they best support the adjacent text.
                • Conclusions panel (4-5 bullets — stated as claims, not hedges)
                • Future directions panel (3 bullets — specific next experiments)
            - FOOTER (full width): Data · Limitations · Acknowledgements

            Output ONLY valid HTML. No markdown, no explanation, no code fences. Start with <!DOCTYPE html>.

            === RESEARCH SYNTHESIS ===
            {final_result}

            === PLAIN-ENGLISH WRITEUP ===
            {writeup_result}
        """).strip()
    }]

    poster_html, tokens, t_html = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, poster_messages, "POSTER HTML"
    )

    # Strip accidental markdown fences
    poster_html = poster_html.strip()
    if poster_html.startswith("```"):
        poster_html = "\n".join(poster_html.splitlines()[1:])
    if poster_html.endswith("```"):
        poster_html = "\n".join(poster_html.splitlines()[:-1])

    poster_html = sanitize_poster_html(poster_html)

    poster_path = OUTPUT_DIR / f"{timestamp}_poster.html"
    with open(poster_path, "w") as f:
        f.write(poster_html)
    print(f"  → Saved (v1): {poster_path}")

    if not with_critique:
        print(f"  → Skipping critique/revision (use --with-critique to enable)")
        print(f"  → Open in browser, then File → Print → Save as PDF (A0, no margins)")
        _print_poster_timing([("Poster HTML", t_html)])
        return poster_path

    # ── Poster critique (uses stripped text — no JS/CSS — to save tokens) ──────
    print(f"\n  [POSTER CRITIQUE] reviewing narrative and communication...")
    poster_text = strip_html_for_critique(poster_html)
    poster_critique_messages = [{
        "role": "user",
        "content": textwrap.dedent(f"""
            Review the following conference poster for narrative flow, communication quality,
            and scientific accuracy. Focus on whether a first-time viewer can follow the story.

            === POSTER CONTENT (text extracted from HTML) ===
            {poster_text}
        """).strip()
    }]

    poster_critique, tokens, t_critique = call_agent(
        client, CRITIQUE_MODEL, POSTER_CRITIQUE_SYSTEM, poster_critique_messages, "POSTER CRITIQUE"
    )
    save_output(f"{timestamp}_poster_critique.md", poster_critique)
    print_section("POSTER CRITIQUE (preview)", poster_critique)

    # ── Poster revision ────────────────────────────────────────────────────────
    print(f"\n  [POSTER REVISION] revising based on critique...")
    poster_revision_messages = [
        {"role": "user", "content": poster_messages[0]["content"]},
        {"role": "model", "content": poster_html},
        {"role": "user", "content": textwrap.dedent(f"""
            A science communication expert has reviewed the poster:

            === CRITIQUE ===
            {poster_critique}
            ================

            Revise the poster HTML to address all critique points.
            Pay special attention to: narrative flow, heading language, question-answer structure.
            Output ONLY valid HTML. No markdown, no explanation. Start with <!DOCTYPE html>.
        """).strip()}
    ]

    poster_html_v2, tokens, t_revision = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, poster_revision_messages, "POSTER REVISION"
    )

    # Strip fences
    poster_html_v2 = poster_html_v2.strip()
    if poster_html_v2.startswith("```"):
        poster_html_v2 = "\n".join(poster_html_v2.splitlines()[1:])
    if poster_html_v2.endswith("```"):
        poster_html_v2 = "\n".join(poster_html_v2.splitlines()[:-1])

    poster_html_v2 = sanitize_poster_html(poster_html_v2)

    with open(poster_path, "w") as f:
        f.write(poster_html_v2)
    print(f"  → Saved (v2, revised): {poster_path}")
    print(f"  → Open in browser, then File → Print → Save as PDF (A0, no margins)")
    _print_poster_timing([("Poster HTML", t_html), ("Poster Critique", t_critique), ("Poster Revision", t_revision)])
    return poster_path


def _print_poster_timing(steps: list[tuple[str, float]]):
    print(f"\n  {'─'*40}")
    print(f"  {'Step':<25} {'Time':>8}")
    print(f"  {'─'*25} {'─'*8}")
    for name, t in steps:
        print(f"  {name:<25} {t:>6.0f}s")
    total = sum(t for _, t in steps)
    print(f"  {'─'*25} {'─'*8}")
    print(f"  {'POSTER TOTAL':<25} {total:>6.0f}s")


# ── Main workflow ──────────────────────────────────────────────────────────────

def run(n_analyses: int = 2, critique_rounds: int = 2):
    global OUTPUT_DIR
    import time as _time
    client = genai.Client(api_key=API_KEY)
    total_tokens = 0
    step_times: list[tuple[str, float]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = OUTPUT_BASE / timestamp
    OUTPUT_DIR.mkdir(exist_ok=True)
    analyses = []  # stores (title, result) for each analysis
    run_start = _time.time()

    print("\n" + "="*60)
    print("  RESEARCHER — AI-driven research workflow")
    print(f"  Model: {ANALYSIS_MODEL}")
    print(f"  Analyses: {n_analyses}  |  Critique rounds: {critique_rounds}")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n[SETUP] Loading data from {DATA_FILE.name}...")
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
                Here is a raw dataset for analysis.

                {data_summary}

                TASK — Analysis 1 of {n_analyses}:
                You are beginning a multi-part research project. This first analysis establishes
                the central question and findings. Subsequent analyses will build on it.

                1. Propose a specific, testable research question suitable for a bachelor's thesis or higher.
                   Choose something scientifically interesting that will naturally lead to a deeper follow-up.
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
            prompt = textwrap.dedent(f"""
                TASK — Analysis {i} of {n_analyses}:
                The full dataset is in your context from the first message. Do not re-request it.
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

        result, tokens, elapsed = call_agent(
            client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, step_label
        )
        total_tokens += tokens
        step_times.append((step_label, elapsed))
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
                Evaluate it as a coherent body of work — does it tell a compelling scientific story?

                {current_body}

                Provide a numbered critique. Be specific. Cite exact claims that need support or correction.
                Pay particular attention to: narrative coherence between analyses, statistical validity,
                and what would make this poster-worthy for a scientific conference audience.
                End with: VERDICT: [ACCEPT / MINOR REVISION / MAJOR REVISION]
            """).strip()
        }]

        critique_result, tokens, elapsed = call_agent(
            client, CRITIQUE_MODEL, CRITIQUE_SYSTEM, critique_messages, f"CRITIQUE R{round_num}"
        )
        total_tokens += tokens
        step_times.append((f"Critique R{round_num}", elapsed))
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

        revised_body, tokens, elapsed = call_agent(
            client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, f"REVISION R{round_num}"
        )
        total_tokens += tokens
        step_times.append((f"Revision R{round_num}", elapsed))
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
            This will be presented at a scientific conference.

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
            Write for a broad scientific audience — assume no prior context.
        """).strip()
    })

    final_result, tokens, elapsed = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, analysis_messages, "FINAL SYNTHESIS"
    )
    total_tokens += tokens
    step_times.append(("Final Synthesis", elapsed))
    save_output(f"{timestamp}_final_synthesis.md", final_result)
    print_section("FINAL SYNTHESIS (preview)", final_result)

    # ── Plain-English writeup ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PLAIN-ENGLISH WRITEUP")
    print(f"{'─'*60}")

    writeup_messages = [{
        "role": "user",
        "content": textwrap.dedent(f"""
            Write a brief, accessible summary of the following research for a broad scientific audience.
            Assume they are scientists but not specialists in this specific field.

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

    writeup_result, tokens, elapsed = call_agent(
        client, ANALYSIS_MODEL, ANALYSIS_SYSTEM, writeup_messages, "WRITEUP"
    )
    total_tokens += tokens
    step_times.append(("Writeup", elapsed))
    save_output(f"{timestamp}_writeup.md", writeup_result)
    print_section("PLAIN-ENGLISH WRITEUP (preview)", writeup_result)

    poster_path = generate_poster(client, final_result, writeup_result, n_analyses, timestamp,
                                  with_critique=True)
    total_tokens += 0  # token count handled inside generate_poster, logged there

    # ── Summary ───────────────────────────────────────────────────────────────
    total_wall = _time.time() - run_start
    print(f"\n{'='*60}")
    print(f"  WORKFLOW COMPLETE")
    print(f"  Analyses: {n_analyses}  |  Critique rounds: {critique_rounds}")
    print(f"  Total tokens: ~{total_tokens:,}")
    print(f"\n  STEP TIMING")
    print(f"  {'─'*40}")
    print(f"  {'Step':<28} {'Time':>8}")
    print(f"  {'─'*28} {'─'*8}")
    for name, t in step_times:
        print(f"  {name:<28} {t:>6.0f}s")
    print(f"  {'─'*28} {'─'*8}")
    print(f"  {'TOTAL (wall clock)':<28} {total_wall:>6.0f}s")
    print(f"\n  Outputs:")
    for f in sorted(OUTPUT_DIR.glob(f"{timestamp}_*.md")):
        print(f"    {f.name}")
    print("="*60)
    warn_tokens(total_tokens)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-driven multi-analysis research workflow")
    parser.add_argument("--analyses", type=int, default=2, choices=[1, 2, 3],
                        help="Number of chained analyses (default: 2, max: 3)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of critique rounds (default: 2)")
    parser.add_argument("--poster-only", metavar="TIMESTAMP",
                        help="Regenerate poster only from existing outputs (e.g. --poster-only 20260401_170011)")
    parser.add_argument("--with-critique", action="store_true",
                        help="Run poster critique+revision loop (adds 2 extra API calls; on by default for full runs)")
    args = parser.parse_args()

    if args.poster_only:
        # Load existing synthesis and writeup, regenerate poster only
        ts = args.poster_only
        OUTPUT_DIR = OUTPUT_BASE / ts
        synthesis_file = OUTPUT_DIR / f"{ts}_final_synthesis.md"
        writeup_file   = OUTPUT_DIR / f"{ts}_writeup.md"
        if not synthesis_file.exists():
            sys.exit(f"ERROR: {synthesis_file} not found. Check your timestamp.")
        if not writeup_file.exists():
            sys.exit(f"ERROR: {writeup_file} not found.")

        final_result   = synthesis_file.read_text()
        writeup_result = writeup_file.read_text()

        load_dotenv(Path(__file__).parent.parent.parent / ".env")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Infer n_analyses from existing files
        n_analyses = sum(1 for f in OUTPUT_DIR.glob(f"{ts}_analysis_*.md") if f.is_file())
        n_analyses = max(n_analyses, 1)

        generate_poster(client, final_result, writeup_result, n_analyses, ts,
                        with_critique=args.with_critique)
        print(f"\n  Poster regenerated for run: {ts}")
    else:
        run(n_analyses=args.analyses, critique_rounds=args.rounds)
