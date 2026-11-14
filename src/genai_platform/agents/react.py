"""ReAct agent from scratch with step-by-step streaming and self-correction."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterator, Sequence

from genai_platform.agents.tools import Tool, default_tools
from genai_platform.llm.gateway import LLMGateway, Usage, get_gateway

logger = logging.getLogger(__name__)

REACT_PROMPT = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to {max_steps} times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
"""


class ReActAgent:
    """From-scratch ReAct agent implementing Thought -> Action -> Observation reasoning."""

    def __init__(
        self,
        tools: Sequence[Tool] | None = None,
        llm: LLMGateway | None = None,
        max_steps: int = 6,
    ):
        self.tools = list(tools) if tools is not None else default_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = llm or get_gateway()
        self.max_steps = max_steps

    def _render_tool_descriptions(self) -> str:
        lines = []
        for t in self.tools:
            lines.append(f"{t.name}: {t.description}")
        return "\n".join(lines)

    def run_stream(self, question: str) -> Iterator[dict[str, Any]]:
        """
        Stream ReAct events:
        - {"type": "thought", "content": ...}
        - {"type": "action", "tool": ..., "input": ...}
        - {"type": "observation", "content": ...}
        - {"type": "token", "content": ...} (for the final answer text)
        - {"type": "final", "answer": ..., "steps": ..., "usage": ..., "latency_s": ..., "truncated": ...}
        """
        start_time = time.perf_counter()
        tool_names = ", ".join(self.tool_map.keys())
        prompt_text = REACT_PROMPT.format(
            tools=self._render_tool_descriptions(),
            tool_names=tool_names,
            max_steps=self.max_steps,
            question=question,
        )

        conversation_scratchpad = prompt_text
        step = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_answer = ""
        truncated = False

        while step < self.max_steps:
            step += 1
            messages = [{"role": "user", "content": conversation_scratchpad}]

            # Call LLM with stop sequence on Observation:
            llm_res = self.llm.chat(
                messages,
                temperature=0.2,
                stop=["Observation:", "\nObservation:"],
            )
            total_prompt_tokens += llm_res.usage.prompt_tokens
            total_completion_tokens += llm_res.usage.completion_tokens

            output = llm_res.text.strip()

            # Parse Final Answer
            final_match = re.search(r"Final Answer:\s*(.*)", output, re.DOTALL | re.IGNORECASE)
            if final_match:
                # Extract thought if present
                thought_match = re.search(r"Thought:\s*(.*?)(?=\s*Final Answer:|$)", output, re.DOTALL | re.IGNORECASE)
                if thought_match:
                    thought = thought_match.group(1).strip()
                    if thought:
                        yield {"type": "thought", "content": thought, "step": step}

                final_answer = final_match.group(1).strip()
                # Yield tokens for live UI rendering
                for tok in re.split(r"(\s+)", final_answer):
                    if tok:
                        yield {"type": "token", "content": tok}

                break

            # Parse Thought and Action
            thought_match = re.search(r"Thought:\s*(.*?)(?=\s*Action:|$)", output, re.DOTALL | re.IGNORECASE)
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", output, re.IGNORECASE)
            action_input_match = re.search(r"Action Input:\s*(.*?)(?=\s*Observation:|$)", output, re.DOTALL | re.IGNORECASE)

            thought = thought_match.group(1).strip() if thought_match else output
            if thought:
                yield {"type": "thought", "content": thought, "step": step}

            if not action_match or not action_input_match:
                # Self-correction: malformed output fed back as observation
                obs = "Observation: Invalid format. You must respond with either 'Action:' and 'Action Input:', or 'Final Answer:'."
                yield {"type": "observation", "content": obs, "step": step}
                conversation_scratchpad += f"\n{output}\n{obs}\n"
                continue

            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip().strip("\"'")

            yield {
                "type": "action",
                "tool": tool_name,
                "input": tool_input,
                "step": step,
            }

            # Execute tool
            if tool_name not in self.tool_map:
                obs = f"Observation: Tool '{tool_name}' not found. Available tools: {tool_names}."
            else:
                try:
                    tool_fn = self.tool_map[tool_name]
                    raw_obs = tool_fn(tool_input)
                    # Truncate observation to 1500 chars
                    if len(raw_obs) > 1500:
                        raw_obs = raw_obs[:1500] + "... [truncated]"
                    obs = f"Observation: {raw_obs}"
                except Exception as e:
                    obs = f"Observation: Tool error: {e}"

            yield {
                "type": "observation",
                "tool": tool_name,
                "content": obs.replace("Observation: ", ""),
                "step": step,
            }

            conversation_scratchpad += f"\n{output}\n{obs}\n"

        if not final_answer:
            truncated = True
            final_answer = "I could not complete the reasoning within the step limit."
            for tok in final_answer.split(" "):
                yield {"type": "token", "content": tok + " "}

        total_latency = time.perf_counter() - start_time
        usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total=total_prompt_tokens + total_completion_tokens,
        )

        yield {
            "type": "final",
            "answer": final_answer,
            "steps": step,
            "usage": usage,
            "latency_s": total_latency,
            "truncated": truncated,
        }

    def run(self, question: str) -> dict[str, Any]:
        """Synchronously collect all stream events and return result."""
        events = list(self.run_stream(question))
        final_event = events[-1] if events and events[-1].get("type") == "final" else {}
        return {
            "question": question,
            "answer": final_event.get("answer", ""),
            "steps": final_event.get("steps", 0),
            "usage": final_event.get("usage", Usage()),
            "latency_s": final_event.get("latency_s", 0.0),
            "truncated": final_event.get("truncated", False),
            "events": events,
        }
