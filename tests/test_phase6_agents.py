import pytest

from genai_platform.agents.react import ReActAgent
from genai_platform.agents.tools import Tool, default_tools, safe_calculator, safe_wikipedia
from genai_platform.llm.gateway import BaseBackend, LLMGateway, LLMResult, Usage


def test_safe_calculator_whitelist_and_security():
    # Valid mathematical operations
    assert safe_calculator("2 + 2") == "4"
    assert safe_calculator("10 * 5 - 8 / 2") == "46"
    assert safe_calculator("2 ^ 3") == "8"
    assert safe_calculator("sqrt(16) + abs(-5)") == "9"
    assert float(safe_calculator("sin(pi / 2)")) == 1.0

    # Exponent cap > 1000 blocked
    res = safe_calculator("2 ** 1005")
    assert "Exponent too large" in res

    # Code execution / eval attempts rejected
    assert "Disallowed" in safe_calculator("__import__('os').system('ls')")
    assert "Disallowed" in safe_calculator("open('/etc/passwd').read()")


def test_safe_wikipedia_handles_errors():
    # Invalid or non-existent topic should return helpful message, not crash
    res = safe_wikipedia("zzzzzz_non_existent_topic_987654321")
    assert isinstance(res, str)
    assert len(res) > 0


class MockMultiStepAgentLLM(BaseBackend):
    name = "mock_agent"
    model = "mock"

    def __init__(self):
        self.step = 0

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        self.step += 1
        prompt = messages[-1]["content"]

        if self.step == 1:
            # Emits malformed output first to test self-correction
            return LLMResult(
                text="I am unsure what to do, let me think aloud without format.",
                usage=Usage(50, 10, 60),
                provider=self.name,
                model=self.model,
                latency_s=0.05,
            )
        elif self.step == 2:
            # Self-corrects: outputs Thought and Action
            return LLMResult(
                text="Thought: I must use calculator.\nAction: calculator\nAction Input: 10 + 15",
                usage=Usage(80, 20, 100),
                provider=self.name,
                model=self.model,
                latency_s=0.05,
            )
        else:
            # Finishes with final answer
            return LLMResult(
                text="Thought: Calculation done.\nFinal Answer: 25",
                usage=Usage(100, 15, 115),
                provider=self.name,
                model=self.model,
                latency_s=0.05,
            )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield from []


def test_react_agent_execution_and_self_correction():
    tools = [
        Tool(name="calculator", description="math", fn=safe_calculator)
    ]
    llm = LLMGateway(primary=MockMultiStepAgentLLM())
    agent = ReActAgent(tools=tools, llm=llm, max_steps=5)

    events = list(agent.run_stream("What is 10 + 15?"))
    types = [e.get("type") for e in events]

    assert "thought" in types
    assert "action" in types
    assert "observation" in types
    assert "final" in types

    final_event = [e for e in events if e.get("type") == "final"][0]
    assert final_event["answer"] == "25"
    assert final_event["truncated"] is False


def test_react_agent_step_limit_truncation():
    class InfiniteLoopLLM(BaseBackend):
        name = "loop"
        model = "loop"
        def chat(self, messages, **kwargs):
            return LLMResult(
                text="Thought: Thinking...\nAction: calculator\nAction Input: 1 + 1",
                usage=Usage(50, 10, 60),
                provider=self.name,
                model=self.model,
                latency_s=0.05,
            )
        def stream(self, messages, **kwargs):
            yield from []

    agent = ReActAgent(tools=[Tool("calculator", "math", safe_calculator)], llm=LLMGateway(InfiniteLoopLLM()), max_steps=3)
    res = agent.run("Stuck question")
    assert res["truncated"] is True
    assert "step limit" in res["answer"].lower()
