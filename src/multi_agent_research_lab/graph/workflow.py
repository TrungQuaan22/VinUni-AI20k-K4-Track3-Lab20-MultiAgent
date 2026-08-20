"""LangGraph workflow implementation."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and executes the multi-agent graph workflow using LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_app: Any = None

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node.supervisor", {"iteration": state.iteration}):
            return self.supervisor.run(state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node.researcher"):
            return self.researcher.run(state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node.analyst"):
            return self.analyst.run(state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node.writer"):
            return self.writer.run(state)

    def _critic_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node.critic"):
            return self.critic.run(state)

    def _route_condition(self, state: ResearchState) -> str:
        """Read the latest routing decision from state."""
        if not state.route_history:
            return END
        last_route = state.route_history[-1]
        if last_route in ("done", "stop", "end"):
            return END
        if last_route in ("researcher", "analyst", "writer", "critic"):
            return last_route
        return END

    def build(self) -> Any:
        """Create and compile the LangGraph workflow."""
        workflow = StateGraph(ResearchState)

        # Add agent nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("critic", self._critic_node)

        # Set entry point to Supervisor
        workflow.set_entry_point("supervisor")

        # Dynamic conditional edge based on supervisor route decision
        workflow.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # Workers hand control back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        return workflow

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the updated state."""
        if self._compiled_app is None:
            graph = self.build()
            self._compiled_app = graph.compile()

        with trace_span("workflow.run", {"query": state.request.query}):
            try:
                result = self._compiled_app.invoke(state)
                if isinstance(result, ResearchState):
                    return result
                if isinstance(result, dict):
                    return ResearchState.model_validate(result)
                return state
            except Exception as exc:
                # Fallback orchestration loop if graph invocation encounters runtime issues
                state.errors.append(f"LangGraph execution fallback: {exc}")
                while state.iteration < self.settings.max_iterations:
                    self.supervisor.run(state)
                    last_route = state.route_history[-1]
                    if last_route == "done":
                        break
                    elif last_route == "researcher":
                        self.researcher.run(state)
                    elif last_route == "analyst":
                        self.analyst.run(state)
                    elif last_route == "writer":
                        self.writer.run(state)
                    elif last_route == "critic":
                        self.critic.run(state)
                    else:
                        break
                return state
