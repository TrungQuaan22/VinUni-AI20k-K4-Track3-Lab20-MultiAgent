"""Benchmark suite for single-agent vs multi-agent evaluation."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Compute the fraction of gathered sources cited in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer_lower = state.final_answer.lower()
    cited_count = 0

    for i, source in enumerate(state.sources, 1):
        idx_pattern = rf"\[{i}\]"
        title_words = [
            w for w in re.sub(r"[^\w\s]", "", source.title.lower()).split() if len(w) > 3
        ]
        title_matched = any(w in answer_lower for w in title_words) if title_words else False
        url_matched = source.url is not None and source.url.lower() in answer_lower
        index_matched = bool(re.search(idx_pattern, state.final_answer))

        if index_matched or url_matched or title_matched:
            cited_count += 1

    return round(min(1.0, cited_count / len(state.sources)), 4)


def compute_quality_score(state: ResearchState) -> float:
    """Estimate a quality score from 0.0 to 10.0 based on structural and grounding metrics."""
    if not state.final_answer or len(state.final_answer.strip()) < 50:
        return 2.0

    score = 4.0  # Base score for valid response

    # 1. Length and depth (up to +2.0)
    word_count = len(state.final_answer.split())
    if word_count >= 150:
        score += 2.0
    elif word_count >= 80:
        score += 1.0

    # 2. Structure and organization (headings, bullet points) (up to +2.0)
    if "#" in state.final_answer and ("-" in state.final_answer or "*" in state.final_answer):
        score += 2.0
    elif "#" in state.final_answer or "-" in state.final_answer:
        score += 1.0

    # 3. Citation grounding (up to +2.0)
    citation_cov = compute_citation_coverage(state)
    score += round(citation_cov * 2.0, 2)

    # 4. Penalty for errors
    if state.errors:
        score -= min(3.0, len(state.errors) * 1.0)

    return round(max(0.0, min(10.0, score)), 2)


def compute_total_cost(state: ResearchState) -> float:
    """Calculate the cumulative estimated token cost from agent results."""
    total = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost is not None and isinstance(cost, (int, float)):
            total += float(cost)
    return round(total, 6)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner, measure performance metrics, and return benchmark result."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    coverage = compute_citation_coverage(state)
    quality = compute_quality_score(state)
    cost = compute_total_cost(state)
    failure = 1.0 if (state.errors or not state.final_answer) else 0.0

    notes_parts: list[str] = []
    if state.route_history:
        notes_parts.append(f"routes={'>'.join(state.route_history)}")
    if state.sources:
        notes_parts.append(f"sources={len(state.sources)}")
    notes = "; ".join(notes_parts) if notes_parts else "single pass"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost if cost > 0 else None,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure,
        notes=notes,
    )
    return state, metrics
