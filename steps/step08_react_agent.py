#!/usr/bin/env python3
"""Step 08: ReAct Agent Reasoning with Safe AST Calculator Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.agents.react import ReActAgent
from genai_platform.agents.tools import Tool, safe_calculator
from genai_platform.llm.gateway import BaseBackend, LLMGateway, LLMResult, Usage


class MockAgentLLM(BaseBackend):
    name = "mock_agent_llm"
    model = "mock-react"

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        self.call_count += 1
        prompt = messages[-1]["content"]

        if "Observation:" not in prompt:
            # Step 1: Decide to use calculator
            return LLMResult(
                text=(
                    "Thought: I need to calculate 25 * 4 + sqrt(144).\n"
                    "Action: calculator\n"
                    "Action Input: 25 * 4 + sqrt(144)"
                ),
                usage=Usage(prompt_tokens=80, completion_tokens=30, total=110),
                provider=self.name,
                model=self.model,
                latency_s=0.1,
            )
        else:
            # Step 2: Conclude with final answer
            return LLMResult(
                text=(
                    "Thought: The calculation returned 112.\n"
                    "Final Answer: The result of 25 * 4 + sqrt(144) is 112."
                ),
                usage=Usage(prompt_tokens=120, completion_tokens=25, total=145),
                provider=self.name,
                model=self.model,
                latency_s=0.1,
            )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield from []


def main():
    print("=== Step 08: ReAct Agent Reasoning Demo ===")

    tools = [
        Tool(
            name="calculator",
            description="Perform mathematical calculations safely via AST.",
            fn=safe_calculator,
        )
    ]
    llm = LLMGateway(primary=MockAgentLLM())
    agent = ReActAgent(tools=tools, llm=llm, max_steps=4)

    question = "Calculate 25 * 4 + sqrt(144)"
    print(f"User Query: {question}\n")

    print("ReAct Step Execution:")
    for event in agent.run_stream(question):
        ev_type = event.get("type")
        if ev_type == "thought":
            print(f"  🧠 [Thought]: {event['content']}")
        elif ev_type == "action":
            print(f"  🛠️ [Action]: {event['tool']} with input '{event['input']}'")
        elif ev_type == "observation":
            print(f"  👁️ [Observation]: {event['content']}")
        elif ev_type == "final":
            print(f"\n🎉 [Final Answer]: {event['answer']}")
            print(f"   Steps: {event['steps']} | Latency: {event['latency_s']:.2f}s")

    print("\nStep 08 completed successfully!")


if __name__ == "__main__":
    main()
