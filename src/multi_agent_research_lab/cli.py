"""Command-line entrypoint for the multi-agent research lab."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent(query_text: str) -> ResearchState:
    """Execute single-agent baseline: single LLM call without external search."""
    request = _parse_query(query_text)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a standalone research assistant. Write a concise, helpful summary "
        "answering the user's research query directly."
    )
    user_prompt = f"Query: {query_text}\n\nAudience: {request.audience}"
    response = llm.complete(system_prompt, user_prompt)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def run_multi_agent(query_text: str) -> ResearchState:
    """Execute multi-agent workflow."""
    state = ResearchState(request=_parse_query(query_text))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent LLM baseline."""
    _init()
    console.print(f"[bold blue]Running Single-Agent Baseline for:[/bold blue] {query}")
    state = run_single_agent(query)
    console.print(
        Panel(
            state.final_answer or "No answer produced.",
            title="Single-Agent Baseline Output",
            style="green",
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent research workflow."""
    _init()
    console.print(f"[bold green]Running Multi-Agent Workflow for:[/bold green] {query}")
    result = run_multi_agent(query)

    console.print(
        Panel(
            result.final_answer or "No answer produced.",
            title="Multi-Agent Final Answer",
            style="cyan",
        )
    )

    # Display routing execution summary table
    table = Table(title="Execution Summary")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="yellow")
    table.add_row("Total Iterations", str(result.iteration))
    table.add_row("Route History", " -> ".join(result.route_history))
    table.add_row("Sources Retrieved", str(len(result.sources)))
    table.add_row("Agent Artifacts", str(len(result.agent_results)))
    table.add_row("Errors Encountered", str(len(result.errors)))
    console.print(table)


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Research query for benchmark",
        ),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output markdown path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run comparison benchmark between single-agent baseline and multi-agent workflow."""
    _init()
    console.print(f"[bold yellow]Running Comparative Benchmark on:[/bold yellow] {query}")

    st_single, m_single = run_benchmark("single_agent", query, run_single_agent)
    st_multi, m_multi = run_benchmark("multi_agent", query, run_multi_agent)

    metrics_list = [m_single, m_multi]
    report_md = render_markdown_report(metrics_list)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    console.print(Panel(report_md, title="Benchmark Report Preview", style="bold green"))
    console.print(f"[bold green]Saved benchmark report to:[/bold green] {out_path}")


if __name__ == "__main__":
    app()
