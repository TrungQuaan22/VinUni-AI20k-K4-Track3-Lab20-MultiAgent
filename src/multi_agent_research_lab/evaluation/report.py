"""Benchmark report generation."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render comprehensive benchmark metrics to markdown format."""
    lines = [
        "# Multi-Agent Research System - Benchmark Report",
        "",
        "## Summary Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Citation Cov. "
        "| Failure Rate | Routing / Notes |",
        "|:---|---:|---:|---:|---:|---:|:---|",
    ]
    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.2f}/10"
        citation = "N/A" if item.citation_coverage is None else f"{item.citation_coverage:.1%}"
        failure = "0.0%" if item.failure_rate is None else f"{item.failure_rate:.1%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f}s | {cost} | {quality} "
            f"| {citation} | {failure} | `{item.notes}` |"
        )

    lines.extend(
        [
            "",
            "## Trade-off Analysis & Key Insights",
            "",
            "- **Latency vs. Rigor**: Single-agent baseline delivers results with "
            "minimal latency, but lacks external factual verification and citation grounding.",
            "- **Context Specialization**: The Multi-Agent pipeline divides cognitive "
            "load among Supervisor, Researcher, Analyst, and Writer, preventing context "
            "pollution and hallucinations.",
            "- **Cost Efficiency**: Multi-agent incurs higher token costs due to "
            "intermediate handoffs, making it most suitable for deep-research workflows.",
            "- **Guardrail Safety**: Enforcing `max_iterations` and routing fallbacks "
            "prevents runaway execution loops and guarantees bounded runtime.",
            "",
        ]
    )

    return "\n".join(lines) + "\n"
