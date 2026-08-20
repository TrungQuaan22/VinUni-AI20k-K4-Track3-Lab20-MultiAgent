"""Critic agent implementation for fact-checking and quality assurance."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking, citation verification, and quality-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to state."""
        if not state.final_answer:
            state.errors.append("CriticAgent: No final answer available to review.")
            return state

        sources_summary = "\n".join(
            f"[{i}] {s.title} ({s.url or 'no url'}): {s.snippet}"
            for i, s in enumerate(state.sources, 1)
        )

        system_prompt = (
            "You are a rigorous peer-review critic. Evaluate the research report:\n"
            "1. Check if citations properly support claims without hallucinations.\n"
            "2. Check if the report comprehensively addresses the user query.\n"
            "3. Assess clarity, structure, and technical depth.\n"
            "Provide a concise review summary and a numeric score from 1.0 to 10.0."
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{sources_summary}\n\n"
            f"Report:\n{state.final_answer}\n\n"
            "Evaluate the report."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "critic.done",
                {
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as exc:
            err_msg = f"CriticAgent failed: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("critic.error", {"error": str(exc)})

        return state
