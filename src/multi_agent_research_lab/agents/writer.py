"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces the final polished research report with explicit citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with grounded synthesis and bibliography."""
        sources_list = []
        for i, s in enumerate(state.sources, 1):
            url_part = f" ({s.url})" if s.url else ""
            sources_list.append(f"[{i}] {s.title}{url_part}")
        sources_text = "\n".join(sources_list) if sources_list else "No explicit source references."

        analysis_context = (
            state.analysis_notes or state.research_notes or "No analysis notes available."
        )

        system_prompt = (
            "You are a technical writer. Write a comprehensive, rigorous research report. "
            "You MUST ground your statements in the provided analysis and sources. "
            "Use numeric inline citations [1], [2], etc., corresponding to the sources. "
            "Include a References section at the end listing every cited source."
        )

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analysis Notes:\n{analysis_context}\n\n"
            f"Available Sources for Citation:\n{sources_text}\n\n"
            "Write an authoritative final report formatted with Markdown headings, "
            "bullet points, comparative insights, and explicit inline citations."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
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
            state.add_trace_event(
                "writer.done",
                {
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as exc:
            err_msg = f"WriterAgent failed: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("writer.error", {"error": str(exc)})

        return state
