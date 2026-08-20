"""Tracing hooks and observability integrations.

Supports standard span context tracking and optional LangSmith / OpenTelemetry tracing.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def setup_tracing(settings: Settings | None = None) -> None:
    """Initialize environment variables for external tracing providers if configured."""
    s = settings or get_settings()
    if s.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
        logger.info("LangSmith tracing enabled for project %s", s.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager recording execution span duration and metadata."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "success",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("Trace span completed: %s in %.4fs", name, span["duration_seconds"])
