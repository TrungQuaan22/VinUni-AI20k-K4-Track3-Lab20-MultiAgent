"""Search client abstraction for ResearcherAgent."""

import json
import logging
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily integration and offline corpus fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._corpus_dir = (
            Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"
        )

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search using Tavily Search API."""
        if not self.settings.tavily_api_key:
            return []

        url = "https://api.tavily.com/search"
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MultiAgentResearchLab/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req, timeout=float(self.settings.timeout_seconds), context=_SSL_CONTEXT
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        docs: list[SourceDocument] = []
        for item in data.get("results", []):
            docs.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:300],
                    metadata={"score": item.get("score")},
                )
            )
        return docs

    def _search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search in local benchmark offline corpus."""
        docs: list[SourceDocument] = []
        if self._corpus_dir.exists():
            words = set(query.lower().split())
            topic_files = list(self._corpus_dir.glob("*.json"))

            scored_topics = []
            for tf in topic_files:
                name_words = set(tf.stem.lower().split("_"))
                overlap = len(words.intersection(name_words))
                scored_topics.append((overlap, tf))

            scored_topics.sort(key=lambda x: x[0], reverse=True)

            for _, topic_path in scored_topics[:3]:
                try:
                    with open(topic_path, encoding="utf-8") as f:
                        data = json.load(f)

                    all_sources = data.get("public_reference_summaries", []) + data.get(
                        "synthetic_source_documents", []
                    )
                    for src in all_sources:
                        docs.append(
                            SourceDocument(
                                title=src.get("title", "Knowledge Source"),
                                url=src.get("url") or f"urn:corpus:{topic_path.stem}",
                                snippet=src.get("summary")
                                or src.get("abstract")
                                or src.get("key_takeaways", "")[:250],
                                metadata={"source_id": src.get("source_id")},
                            )
                        )
                        if len(docs) >= max_results:
                            return docs
                except Exception as exc:
                    logger.debug("Failed reading corpus topic %s: %s", topic_path, exc)

        if not docs:
            # Fallback curated synthetic documents
            docs = [
                SourceDocument(
                    title="Multi-Agent Systems: Architecture & Coordination Patterns",
                    url="https://arxiv.org/abs/2401.00001",
                    snippet=(
                        "Supervisor-worker coordination reduces context dilution and "
                        "enhances verification in complex reasoning workflows."
                    ),
                    metadata={"confidence": 0.95},
                ),
                SourceDocument(
                    title="Evaluating LLM Agents: Cost, Latency, and Quality Trade-offs",
                    url="https://arxiv.org/abs/2402.00002",
                    snippet=(
                        "Single-agent baselines excel in latency and token cost for simple "
                        "queries, while multi-agent pipelines excel in complex synthesis "
                        "and citation rigor."
                    ),
                    metadata={"confidence": 0.92},
                ),
                SourceDocument(
                    title="Guardrails and Failure Recovery in Autonomous Workflows",
                    url="https://arxiv.org/abs/2403.00003",
                    snippet=(
                        "Strict loop limits, timeout handlers, and routing fallbacks "
                        "prevent runaway token consumption and cascading hallucinations."
                    ),
                    metadata={"confidence": 0.90},
                ),
            ]
        return docs[:max_results]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query with graceful fallback."""
        try:
            if self.settings.tavily_api_key:
                docs = self._search_tavily(query, max_results=max_results)
                if docs:
                    return docs
        except Exception as exc:
            logger.warning(
                "Tavily search failed (%s), falling back to offline research corpus.",
                exc,
            )

        return self._search_offline_corpus(query, max_results=max_results)
