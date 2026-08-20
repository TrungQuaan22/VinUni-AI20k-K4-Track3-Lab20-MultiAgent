"""Supervisor / router implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def route(self, state: ResearchState) -> str:
        """Determine next agent or completion state based on shared state."""
        # 1. Guardrail against runaway iterations
        if state.iteration >= self.settings.max_iterations:
            return "done"

        # 2. Check if final answer already produced
        if state.final_answer:
            return "done"

        # 3. Missing sources/research notes -> Researcher
        if not state.sources and not state.research_notes:
            return "researcher"

        # 4. Missing analysis -> Analyst
        if not state.analysis_notes:
            return "analyst"

        # 5. Missing final synthesis -> Writer
        if not state.final_answer:
            return "writer"

        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, determine next route, and record decision in state."""
        next_route = self.route(state)
        state.record_route(next_route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routing decision: {next_route} (iteration {state.iteration})",
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        state.add_trace_event(
            "supervisor.route", {"next": next_route, "iteration": state.iteration}
        )
        return state
