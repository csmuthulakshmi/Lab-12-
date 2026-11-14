"""CrewAI multi-agent report generation: Researcher -> Writer -> Reviewer."""

from __future__ import annotations

import logging
from typing import Any

from genai_platform.config import get_settings
from genai_platform.llm.gateway import get_gateway
from genai_platform.llm.lc_model import GatewayChatModel

logger = logging.getLogger(__name__)


def _build_crew_llm() -> Any:
    """Build the LLM instance for CrewAI agents."""
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    # Try passing GatewayChatModel or provider string
    try:
        return GatewayChatModel()
    except Exception:
        if provider == "gemini":
            return f"gemini/{settings.GEMINI_MODEL}"
        elif provider == "ollama":
            return f"ollama/{settings.OLLAMA_MODEL}"
        return f"openai/{settings.OPENAI_MODEL}"


def run_report(retriever: Any, topic: str) -> dict[str, Any]:
    """
    Run a sequential multi-agent Crew (Researcher -> Writer -> Reviewer)
    to research, synthesize, and fact-check a report on the given topic.
    Lazy imports crewai to ensure fast startup and clean testability.
    """
    try:
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import tool
    except ImportError as e:
        logger.warning(f"CrewAI import failed: {e}. Returning mock report structure.")
        return {
            "topic": topic,
            "report": f"Report on '{topic}' (CrewAI unavailable):\n\nKey findings based on available documents.",
            "task_outputs": [
                {"task": "research", "output": f"Notes on {topic}"},
                {"task": "writing", "output": f"Draft on {topic}"},
                {"task": "review", "output": f"Fact-checked report on {topic}"},
            ],
        }

    # Define tool for searching the knowledge base
    @tool("kb_search")
    def kb_search(query: str) -> str:
        """Search the knowledge base for documents and context on a specific topic."""
        hits = retriever.retrieve(query, k=5)
        if not hits:
            return "No relevant documents found."
        lines = []
        for i, h in enumerate(hits, start=1):
            lines.append(f"Passage [{i}] ({h.cite}):\n{h.text}")
        return "\n\n".join(lines)

    llm = _build_crew_llm()

    # 1. Researcher
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Thoroughly research and compile factual evidence from the knowledge base about '{topic}'",
        backstory=(
            "You are a meticulous research analyst skilled at querying knowledge bases, "
            "verifying sources, and summarizing raw findings into structured notes."
        ),
        tools=[kb_search],
        llm=llm,
        verbose=False,
    )

    # 2. Writer
    writer = Agent(
        role="Technical Content Specialist",
        goal=f"Synthesize research notes into a coherent, professional report on '{topic}'",
        backstory=(
            "You are an expert technical communicator known for clear explanations, "
            "logical structuring, and retaining technical precision."
        ),
        llm=llm,
        verbose=False,
    )

    # 3. Reviewer
    reviewer = Agent(
        role="Chief Fact-Checking Editor",
        goal=f"Fact-check the draft report against research notes and finalize the output for '{topic}'",
        backstory=(
            "You are a relentless editor who fact-checks every claim, removes ungrounded speculation, "
            "and appends clear 'Review notes' confirming verification."
        ),
        llm=llm,
        verbose=False,
    )

    # Define sequential tasks
    task_research = Task(
        description=f"Search the knowledge base for information on '{topic}'. Compile detailed factual notes with source citations.",
        expected_output="A structured bullet-point list of research findings and source citations.",
        agent=researcher,
    )

    task_write = Task(
        description=f"Draft a comprehensive technical report on '{topic}' using only the researcher's findings.",
        expected_output="A well-organized, comprehensive draft report with introduction, key sections, and conclusion.",
        agent=writer,
    )

    task_review = Task(
        description=(
            f"Fact-check every assertion in the draft report against the research notes for '{topic}'. "
            "Output the final verified report followed by a '## Review notes' section documenting verification status."
        ),
        expected_output="Final fact-checked report with a dedicated Review notes section.",
        agent=reviewer,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[task_research, task_write, task_review],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        report_text = str(result)
        task_outputs = [
            {"task": "research", "output": str(task_research.output.raw if hasattr(task_research, "output") and task_research.output else "")},
            {"task": "writing", "output": str(task_write.output.raw if hasattr(task_write, "output") and task_write.output else "")},
            {"task": "review", "output": str(task_review.output.raw if hasattr(task_review, "output") and task_review.output else "")},
        ]
        return {
            "topic": topic,
            "report": report_text,
            "task_outputs": task_outputs,
        }
    except Exception as e:
        logger.error(f"Crew execution failed: {e}")
        return {
            "topic": topic,
            "report": f"Crew execution error: {e}",
            "task_outputs": [],
        }
