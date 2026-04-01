# Researcher — AI-Driven Research Project

## Concept

Feed a raw, unanalysed dataset to an AI model (target: GPT or equivalent frontier model) and have it autonomously:

1. **Propose a project** — generate a bachelor's thesis-level (or higher) research question and study design based solely on the data
2. **Analyse the data** — perform analysis aligned with the proposed project (stats, plots, patterns)
3. **Report results** — produce a structured results document (methods, findings, figures)
4. **Make a poster** — convert the results document into a conference-ready research poster

## Workflow

```
raw data
   └── AI: propose project idea + study design
          └── AI: analyse data
                 └── AI: write results document
                        └── AI: generate research poster
```

## Key Questions This Demo Explores

- How well can a frontier model formulate a meaningful scientific question from raw data?
- Can it conduct statistically appropriate analysis without hand-holding?
- Does the output meet a real academic standard, or does it hallucinate/cut corners?
- What are the failure modes?

## Files (to be added)

- `data/` — raw input dataset(s)
- `project_proposal.md` — AI-generated thesis proposal
- `analysis/` — code and outputs from the analysis step
- `results.md` — AI-generated results document
- `poster/` — final conference poster (PDF or image)
- `reflection.md` — commentary on what the AI got right/wrong

## Notes for Claude

- The user will supply raw data — help identify what type of analysis is appropriate
- The AI doing the research is a separate frontier model (e.g. GPT); Claude's role here is to help scaffold, review, and critique the outputs
- Quality bar: bachelor's thesis minimum, ideally publishable-standard
- Document failures and hallucinations — they are part of the demo's story
- The poster should be visually presentable at a real conference (layout, figures, concise text)
