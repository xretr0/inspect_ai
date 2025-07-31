from typing import Any
from collections.abc import Callable

from inspect_ai._util.content import (
    ContentAudio,
    ContentImage,
    ContentVideo,
)
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import (
    ToolChoice,
    ToolInfo,
)


class CustomAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
        )

        # Get the generate function from external code that initialized it
        assert "generate_function" in model_args, "generate_function is required in model_args"
        assert isinstance(model_args["generate_function"], Callable), "generate_function must be a of type Callable[[str, GenerateConfig], dict[str, Any]]"
        self.generate_function: Callable[[str, GenerateConfig], dict[str, Any]] = model_args["generate_function"]

        # This lets the user specify the way the chat history is converted into a string
        self.chat_history_parser: Callable[[list[ChatMessage]], str] = model_args.get("chat_history_parser", message_content_to_string)
        assert isinstance(self.chat_history_parser, Callable), "chat_history_parser must be a of type Callable[[list[ChatMessage]], str]"


    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise NotImplementedError("Tools are not yet supported for custom provider")

        input_str = self.chat_history_parser(input)

        output = self.generate_function(input_str, config)
        assert isinstance(output, dict), "generate_function must return a dictionary"
        assert "response" in output, "generate_function must contain a 'response' key with the generated response"
        assert isinstance(output["response"], str), "generate_function's response must be a string"

        response: str = output["response"]
        input_tokens: int = output.get("input_tokens", 0)
        output_tokens: int = output.get("output_tokens", 0)
        total_tokens: int = input_tokens + output_tokens
        time: float | None = output.get("time", None)
        metadata: dict[str, Any] | None = output.get("metadata", None)
        error: str | None = output.get("error", None)

        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(
                content=response,
                model=self.model_name,
                source="generate",
            ),
        )

        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ) if total_tokens > 0 else None,
            time=time,
            metadata=metadata,
            error=error,
        )


def message_content_to_string(messages: list[ChatMessage]) -> str:
    """Convert list of content in `ChatMessageAssistant`, `ChatMessageUser` or `ChatMessageSystem` to a string.

    This is the default chat history parser for the custom provider.
    Modified from the HuggingFace provider.
    """
    out = ""
    for message in messages:
        if isinstance(message.content, list):
            is_multimodal = any(
                isinstance(item, ContentAudio | ContentImage | ContentVideo)
                for item in message.content
            )
            if is_multimodal:
                raise NotImplementedError(
                    "Custom provider does not support multimodal content, please provide text inputs only."
                )
            message.content = message.text
        out += f"{message.role}: {message.content}\n"

    return out
