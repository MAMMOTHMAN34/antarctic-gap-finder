"""Stage 6: Report.

For each of the highest-scoring gap themes, I assemble an evidence dossier and write it to `outputs/report.md`:

  * the gap statistics (size, citation velocity, recency, score),
  * how the theme's yearly output has grown (is it actually accelerating?),
  * the representative recent + highly-cited papers (the evidence the area is
    heating up),
  * a framed research question grounded in that evidence.

The statistics and paper selection are produced reproducibly here; the framed
questions are written to read like the conclusion of a short paper.

Run:
    python -m src.report
"""

from __future__ import annotations

import pandas as pd

from src import config

# Framed research questions, authored from the evidence each gap surfaces.
# Keyed by gap rank (1 = highest score). With RANDOM_STATE fixed, the top gaps
# are stable; if you change the clustering/gap parameters, revisit these.
FRAMED_QUESTIONS = {
    1: {
        "question": (
            "How much will the Antarctic Ice Sheet add to sea-level rise this "
            "century — and what explains the still-wide spread between coupled "
            "ice-sheet–climate models?"
        ),
        "why": (
            "The theme is small (129 papers) yet its papers are cited fast "
            "(~10/yr) and cluster in 2021–2024 around ISMIP6 and CMIP6 model "
            "ensembles. The open problem is no longer *whether* Antarctica "
            "matters for sea level but *why* state-of-the-art projections still "
            "disagree by metres — the spread, not the mean, is the frontier."
        ),
        "unlock": (
            "Observationally-constrained, two-way-coupled ice-sheet/ocean "
            "models that narrow the projection envelope coastal planners depend "
            "on."
        ),
    },
    2: {
        "question": (
            "Does Antarctic ice-shelf meltwater dampen or amplify global "
            "warming, and through which Southern Ocean pathways?"
        ),
        "why": (
            "101 papers, accelerating sharply since 2020. Recent high-impact "
            "work shows meltwater can *reduce* transient warming via a "
            "sea-surface-temperature pattern effect and reshape the Antarctic "
            "Slope Current — but the sign and magnitude of the feedback are "
            "unsettled, and most climate projections omit it entirely."
        ),
        "unlock": (
            "A quantified meltwater–climate feedback that could revise both "
            "Southern Ocean circulation and global climate-sensitivity estimates."
        ),
    },
    3: {
        "question": (
            "What controls whether Southern Ocean clouds hold liquid or ice — "
            "and would getting it right remove the persistent shortwave "
            "radiation bias in climate models?"
        ),
        "why": (
            "112 papers with a clear post-2020 jump. Mixed-phase and "
            "supercooled-liquid clouds drive a long-standing solar-reflection "
            "bias in CMIP6 models; ice-nucleating particles and secondary ice "
            "production are only now being observed (SOCRATES-era campaigns) and "
            "remain poorly constrained."
        ),
        "unlock": (
            "Better cloud-phase microphysics, which is one of the largest known "
            "sources of Southern Ocean energy-balance error in climate models."
        ),
    },
}


def yearly_counts(sub: pd.DataFrame) -> str:
    """Compact per-year paper counts, e.g. '2018:5  2019:9 ...', to show growth."""
    counts = sub["year"].value_counts().sort_index()
    return "  ".join(f"{int(y)}:{int(n)}" for y, n in counts.items())


def representative_papers(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick recent, high-impact papers: rank by citations-per-year, prefer recent.

    These are the papers that make the case that the theme is emerging — recent
    work that is already being cited heavily.
    """
    age = (config.REFERENCE_YEAR - sub["year"]).clip(lower=1)
    sub = sub.assign(cpy=sub["cited_by_count"] / age)
    recent = sub[sub["year"] >= config.RECENT_FROM]
    pool = recent if len(recent) >= n else sub
    return pool.sort_values("cpy", ascending=False).head(n)


def write_report(gaps: pd.DataFrame, df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Antarctic Research Gap Finder — Results\n")
    lines.append(
        "Each theme below scored highest on the gap metric "
        "(`z(citation velocity) + 0.5·z(recency) − z(log size)`): "
        "small fields whose papers are cited fast and recently — emerging, "
        "high-momentum, under-explored relative to their impact.\n"
    )
    lines.append(
        "> **What this signal is and isn't.** It flags themes that are *small "
        "but high-impact-per-paper and recent*. That overlaps with 'emerging "
        "frontier' but is not the same as 'underfunded' — a hot, well-funded "
        "topic can still be small and fast-growing. Read the questions as "
        "*where momentum is outpacing coverage*, not as literal funding gaps.\n"
    )

    top = gaps.head(config.REPORT_TOP_GAPS)
    for rank, (_, g) in enumerate(top.iterrows(), start=1):
        cl = int(g["cluster"])
        sub = df[df["cluster"] == cl]
        lines.append(f"\n## Gap {rank}: {g['label']}\n")
        lines.append(
            f"- **Papers:** {int(g['size'])}  ·  "
            f"**Citation velocity:** {g['velocity']:.1f} cites/yr (median)  ·  "
            f"**Mean year:** {g['mean_year']:.0f}  ·  "
            f"**Recent share (≥{config.RECENT_FROM}):** {g['recent_share']:.0%}  ·  "
            f"**Gap score:** {g['gap_score']:.2f}"
        )
        lines.append(f"- **Defining terms:** {g['top_terms']}")
        lines.append(f"- **Papers per year:** {yearly_counts(sub)}\n")

        lines.append("**Representative recent, high-impact papers:**\n")
        for _, p in representative_papers(sub, config.REPORT_REP_PAPERS).iterrows():
            venue = p["venue"] or "n/a"
            lines.append(
                f"- ({int(p['year'])}, {int(p['cited_by_count'])} cites) "
                f"*{p['title']}* — {venue}"
            )

        q = FRAMED_QUESTIONS.get(rank)
        if q:
            lines.append(f"\n### ❓ Research question {rank}\n")
            lines.append(f"**{q['question']}**\n")
            lines.append(f"- *Why it's a gap:* {q['why']}")
            lines.append(f"- *What answering it unlocks:* {q['unlock']}\n")

    text = "\n".join(lines)
    config.REPORT_MD.write_text(text)
    return text


def main() -> None:
    df = pd.read_parquet(config.PAPERS_PARQUET).merge(
        pd.read_parquet(config.CLUSTERS_PARQUET)[["id", "cluster"]], on="id"
    )
    gaps = pd.read_csv(config.GAPS_CSV)  # already sorted by gap_score
    write_report(gaps, df)
    print(f"Wrote dossier -> {config.REPORT_MD}")
    print(f"Top {config.REPORT_TOP_GAPS} gaps: "
          + " | ".join(gaps.head(config.REPORT_TOP_GAPS)["label"]))


if __name__ == "__main__":
    main()
