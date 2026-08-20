"""Unit tests for agents, services, and multi-agent workflow."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    run_benchmark,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_supervisor_routing_policy() -> None:
    settings = Settings(max_iterations=6)
    supervisor = SupervisorAgent(settings=settings)

    state = ResearchState(request=ResearchQuery(query="Test multi-agent systems"))

    # Initial state -> routes to researcher
    assert supervisor.route(state) == "researcher"
    state = supervisor.run(state)
    assert state.route_history == ["researcher"]
    assert state.iteration == 1

    # Has sources -> routes to analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    assert supervisor.route(state) == "analyst"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst"]

    # Has analysis -> routes to writer
    state.analysis_notes = "Analyzed points."
    assert supervisor.route(state) == "writer"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer"]

    # Has final answer -> done
    state.final_answer = "Final report."
    assert supervisor.route(state) == "done"
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_max_iterations_guardrail() -> None:
    settings = Settings(max_iterations=3)
    supervisor = SupervisorAgent(settings=settings)
    state = ResearchState(request=ResearchQuery(query="Test guardrail"))
    state.iteration = 3
    assert supervisor.route(state) == "done"


def test_researcher_agent_run() -> None:
    search_client = SearchClient()
    researcher = ResearcherAgent(search_client=search_client)
    state = ResearchState(request=ResearchQuery(query="AI Agents in production", max_sources=2))

    updated_state = researcher.run(state)
    assert len(updated_state.sources) > 0
    assert updated_state.research_notes is not None
    assert len(updated_state.agent_results) == 1
    assert updated_state.agent_results[0].agent == "researcher"


def test_analyst_agent_run() -> None:
    llm = LLMClient()
    analyst = AnalystAgent(llm_client=llm)
    state = ResearchState(request=ResearchQuery(query="Compare single-agent vs multi-agent"))
    state.sources = [
        SourceDocument(title="Doc 1", snippet="Single agents have lower latency"),
        SourceDocument(title="Doc 2", snippet="Multi-agent systems improve modularity"),
    ]
    updated_state = analyst.run(state)
    assert updated_state.analysis_notes is not None
    assert len(updated_state.agent_results) == 1
    assert updated_state.agent_results[0].agent == "analyst"


def test_writer_agent_run() -> None:
    llm = LLMClient()
    writer = WriterAgent(llm_client=llm)
    state = ResearchState(request=ResearchQuery(query="Summarize GraphRAG"))
    state.sources = [
        SourceDocument(
            title="GraphRAG Paper",
            url="https://arxiv.org/abs/2404.16130",
            snippet="Graph-based RAG",
        ),
    ]
    state.analysis_notes = "GraphRAG builds a knowledge graph index."
    updated_state = writer.run(state)
    assert updated_state.final_answer is not None
    assert len(updated_state.agent_results) == 1
    assert updated_state.agent_results[0].agent == "writer"


def test_critic_agent_run() -> None:
    llm = LLMClient()
    critic = CriticAgent(llm_client=llm)
    state = ResearchState(request=ResearchQuery(query="Summarize GraphRAG"))
    state.sources = [SourceDocument(title="GraphRAG Paper", snippet="Graph-based RAG")]
    state.final_answer = "GraphRAG is a novel method [1]."
    updated_state = critic.run(state)
    assert len(updated_state.agent_results) == 1
    assert updated_state.agent_results[0].agent == "critic"


def test_multi_agent_workflow_end_to_end() -> None:
    workflow = MultiAgentWorkflow()
    state = ResearchState(
        request=ResearchQuery(query="Research GraphRAG state-of-the-art and summarize")
    )
    result = workflow.run(state)
    assert result.final_answer is not None
    assert len(result.sources) > 0
    assert len(result.route_history) >= 3
    assert result.route_history[-1] == "done"


def test_benchmark_metrics() -> None:
    state = ResearchState(request=ResearchQuery(query="Test benchmark metrics"))
    state.sources = [
        SourceDocument(title="GraphRAG", snippet="Graph RAG methodology"),
        SourceDocument(title="Agentic Workflows", snippet="Multi-agent patterns"),
    ]
    state.final_answer = (
        "This report discusses GraphRAG [1] and Agentic Workflows [2] in detail.\n\n"
        "## Section\n- Point 1"
    )

    cov = compute_citation_coverage(state)
    assert cov == 1.0

    quality = compute_quality_score(state)
    assert 0.0 <= quality <= 10.0

    def mock_runner(q: str) -> ResearchState:
        return state

    _, metrics = run_benchmark("test_run", "Test benchmark query", mock_runner)
    assert metrics.run_name == "test_run"
    assert metrics.citation_coverage == 1.0
    assert metrics.quality_score is not None
