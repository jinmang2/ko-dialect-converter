"""Single source of truth for what each leaderboard metric means and which way is good.

Every metric the leaderboard prints is registered here with its direction (↑/↓), the
axis group it belongs to, a one-line meaning, and a caveat. This kills the recurring
"is a low copy_margin good?" confusion (it is NOT — copy_margin is higher-is-better) and
lets aggregation/Pareto code ask ``is_higher_better(metric)`` instead of hard-coding it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    higher_is_better: bool
    group: str  # "selection" | "dialectness" | "surface" | "monitoring"
    meaning: str
    caveat: str = ""

    @property
    def arrow(self) -> str:
        return "↑" if self.higher_is_better else "↓"


# group meanings:
#   selection   — proxy-independent, drives checkpoint/model choice (plan §3 A4)
#   dialectness — how strongly the output adopts the dialect (some are classifier proxies)
#   surface     — n-gram overlap vs gold; copy-biased, read with copy_margin
#   monitoring  — useful to watch, must NOT drive selection (circularity)
_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "reconstruction_bleu",
        True,
        "selection",
        "BLEU of dialect→standard back-translation vs source; content preserved if high",
        "proxy-independent — the ranking metric (Dual-RL / Luo 2019)",
    ),
    MetricSpec(
        "copy_margin",
        True,
        "dialectness",
        "chrF(gen,gold) − chrF(gen,source): >0 means closer to gold dialect than to source",
        "HIGHER is better; negative = output copies the standard input (Mind the Style Gap 2502.15022)",
    ),
    MetricSpec(
        "gain_over_copy",
        True,
        "selection",
        "chrF(gen,gold) − chrF(source,gold): value added over emitting the input unchanged",
        "NEGATIVE = worse than doing nothing. The only surface axis comparable ACROSS "
        "regions, since each region's copy floor differs (56.7 Gyeongsang … 79.9 "
        "Chungcheong … 18.9 Jeju)",
    ),
    MetricSpec(
        "copy_baseline",
        False,
        "surface",
        "chrF(source,gold): what copying the input already scores — a property of the "
        "region, not the model",
        "not a score to maximise; the floor that makes raw chrF comparable",
    ),
    MetricSpec(
        "tdr",
        True,
        "dialectness",
        "Target Dialect Rate: fraction the classifier tags as the target dialect",
        "classifier proxy → monitoring only, not for selection (DIA-REFINE 2511.06680)",
    ),
    MetricSpec(
        "dfs",
        True,
        "dialectness",
        "Dialect Fidelity Score: log-ratio of cos(out,dialect) vs cos(out,standard) embeddings",
        "classifier-embedding proxy → monitoring only (DIA-REFINE 2511.06680 Eq.1)",
    ),
    MetricSpec(
        "chrf",
        True,
        "surface",
        "character-F vs gold dialect",
        "copy-biased where gold≈source (Gangwon) — read with copy_margin",
    ),
    MetricSpec(
        "bleu",
        True,
        "surface",
        "BLEU vs gold dialect",
        "copy-biased — read with copy_margin",
    ),
    MetricSpec(
        "eojeol_accuracy",
        True,
        "dialectness",
        "fraction of expected dialect eojeols appearing verbatim in the output",
        "lexical transformation signal (DIA-REFINE-adjacent)",
    ),
    MetricSpec(
        "chrf_source",
        False,
        "surface",
        "character-F vs the standard source — measures how much the output just copies input",
        "LOWER is better for conversion; only used to derive copy_margin",
    ),
    MetricSpec(
        "jscore",
        True,
        "monitoring",
        "geometric mean of remapped {tdr, copy_margin, eojeol, fluency}",
        "MONITORING ONLY — overlaps training rewards → circular, never select on it (plan §3 A4)",
    ),
)

REGISTRY: dict[str, MetricSpec] = {spec.key: spec for spec in _SPECS}


def get_spec(metric: str) -> MetricSpec:
    """Return the MetricSpec, or a permissive higher-is-better default if unregistered."""
    return REGISTRY.get(
        metric,
        MetricSpec(metric, True, "monitoring", "(unregistered metric)", ""),
    )


def is_higher_better(metric: str) -> bool:
    return get_spec(metric).higher_is_better


def arrow(metric: str) -> str:
    """``↑`` / ``↓`` direction marker for a metric (default ↑ if unregistered)."""
    return get_spec(metric).arrow


def header_label(metric: str) -> str:
    """Column label with its direction arrow, e.g. ``copy_margin↑``."""
    return f"{metric}{arrow(metric)}"


def orient(metric: str, value: float) -> float:
    """Map a metric to a 'higher is better' orientation for ranking/aggregation.

    Higher-is-better metrics pass through; lower-is-better are negated, so callers can
    always ``max``/sort descending without special-casing direction.
    """
    return value if is_higher_better(metric) else -value


def glossary_markdown(metrics: tuple[str, ...] | None = None) -> str:
    """Render a markdown glossary (direction + meaning + caveat) for the given metrics."""
    keys = metrics if metrics is not None else tuple(REGISTRY)
    lines = ["| metric | dir | group | meaning | caveat |", "|---|---|---|---|---|"]
    for k in keys:
        s = get_spec(k)
        better = "higher" if s.higher_is_better else "lower"
        lines.append(f"| `{k}` | {s.arrow} {better} | {s.group} | {s.meaning} | {s.caveat} |")
    return "\n".join(lines)
