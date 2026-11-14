"""LangChain ChatModel adapter wrapping our LLM Gateway."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from pydantic import Field

from genai_platform.llm.gateway import LLMGateway, get_gateway


def _convert_message_to_dict(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": str(message.content)}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": str(message.content)}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": str(message.content)}
    else:
        return {"role": "user", "content": str(message.content)}


class GatewayChatModel(BaseChatModel):
    """BaseChatModel implementation backed by the platform LLMGateway."""

    gateway: LLMGateway = Field(default_factory=get_gateway)
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "genai_gateway"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        dict_messages = [_convert_message_to_dict(m) for m in messages]
        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.max_tokens)

        result = self.gateway.chat(
            messages=dict_messages,
            temperature=temp,
            max_tokens=max_tok,
            stop=stop,
        )

        usage_meta = {
            "input_tokens": result.usage.prompt_tokens,
            "output_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total,
        }

        ai_msg = AIMessage(
            content=result.text,
            usage_metadata=usage_meta,
        )
        generation = ChatGeneration(message=ai_msg)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        dict_messages = [_convert_message_to_dict(m) for m in messages]
        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.max_tokens)

        for chunk in self.gateway.stream(
            messages=dict_messages,
            temperature=temp,
            max_tokens=max_tok,
            stop=stop,
        ):
            if chunk.text:
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                msg_chunk = AIMessageChunk(content=chunk.text)
                yield ChatGenerationChunk(message=msg_chunk)

            if chunk.usage:
                usage_meta = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total,
                }
                msg_chunk = AIMessageChunk(content="", usage_metadata=usage_meta)
                yield ChatGenerationChunk(message=msg_chunk)
