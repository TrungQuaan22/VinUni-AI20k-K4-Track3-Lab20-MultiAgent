"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured analytical insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        context = state.research_notes or "\n".join(
            f"- {d.title}: {d.snippet}" for d in state.sources
        )

        if not context:
            state.errors.append("AnalystAgent: No research notes or sources available.")
            state.add_trace_event("analyst.error", {"error": "no_context"})
            context = f"Topic query: {state.request.query}"

        system_prompt = (
            "You are a senior technical analyst. Analyze gathered research materials, "
            "extract key claims, evaluate evidence strength, contrast viewpoints, "
            "and highlight trade-offs. "
            "Structure your output with clear markdown headings and bullet points."
        )
        user_prompt = (
            f"User Research Query: {state.request.query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Research Evidence:\n{context}\n\n"
            "Provide a structured analytical summary with takeaways and trade-offs."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst.done",
                {
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as exc:
            err_msg = f"AnalystAgent failed: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("analyst.error", {"error": str(exc)})

        return state
