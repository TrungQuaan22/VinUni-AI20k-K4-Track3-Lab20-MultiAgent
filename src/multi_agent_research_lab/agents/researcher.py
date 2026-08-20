"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates structured research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        try:
            docs = self.search_client.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = docs
            notes_lines = [f"### Research Findings ({len(docs)} sources retrieved)"]
            for i, doc in enumerate(docs, 1):
                url_str = f" ({doc.url})" if doc.url else ""
                notes_lines.append(f"{i}. **{doc.title}**{url_str}\n   {doc.snippet}")

            state.research_notes = "\n".join(notes_lines)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes,
                    metadata={"num_sources": len(docs)},
                )
            )
            state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        except Exception as exc:
            err_msg = f"ResearcherAgent failed: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("researcher.error", {"error": str(exc)})
        return state
