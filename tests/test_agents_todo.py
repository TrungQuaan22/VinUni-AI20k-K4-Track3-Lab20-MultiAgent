"""Guard test verifying all components are implemented."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_all_components_implemented_without_todo_errors() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    # Verify clients don't raise StudentTodoError
    llm = LLMClient()
    resp = llm.complete("System prompt", "User prompt")
    assert resp.content

    searcher = SearchClient()
    docs = searcher.search("test query", max_results=2)
    assert isinstance(docs, list)

    # Verify agents don't raise StudentTodoError
    sup = SupervisorAgent()
    state = sup.run(state)
    assert state.route_history

    res_agent = ResearcherAgent(search_client=searcher)
    state = res_agent.run(state)
    assert state.sources

    analyst = AnalystAgent(llm_client=llm)
    state = analyst.run(state)
    assert state.analysis_notes

    writer = WriterAgent(llm_client=llm)
    state = writer.run(state)
    assert state.final_answer

    critic = CriticAgent(llm_client=llm)
    state = critic.run(state)
    assert len(state.agent_results) > 0

    # Verify workflow runs end-to-end
    wf = MultiAgentWorkflow()
    final_st = wf.run(ResearchState(request=ResearchQuery(query="Explain multi-agent systems")))
    assert final_st.final_answer
