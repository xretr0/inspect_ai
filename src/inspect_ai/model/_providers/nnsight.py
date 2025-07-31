from typing import Any
from collections.abc import Callable

from nnsight import LanguageModel  # type: ignore

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


class NNSightAPI(ModelAPI):
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

        # Get the model from external code that initialized it
        assert "nnsight_model" in model_args, "nnsight_model is required in model_args"
        self.model: LanguageModel = model_args["nnsight_model"]
        assert isinstance(self.model, LanguageModel), "nnsight_model must be a nnsight.LanguageModel"

        # Get the hook that stores the activations from the external code
        assert "nnsight_hook" in model_args, "nnsight_hook is required in model_args"
        self.hook: Callable[[str], None] = model_args.get("nnsight_hook", default_hook)
        assert isinstance(self.hook, Callable), "nnsight_hook must be a of type Callable[[str], None]"

        # This lets the user specify the way the chat history is converted into a string
        self.chat_history_parser: Callable[[list[ChatMessage]], str] = model_args.get("chat_history_parser", default_chat_history_parser)
        assert isinstance(self.chat_history_parser, Callable), "chat_history_parser must be a of type Callable[[list[ChatMessage]], str]"

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        input_str = self.chat_history_parser(input)

        max_new_tokens = config.max_tokens or 100  # The default for nnsight is only 1, for convenience we set it to 100

        with self.model.generate(input_str, max_new_tokens=max_new_tokens):
            self.hook(input_str)
            output = self.model.generator.output.save()
        
        input_token_length = len(self.model.tokenizer.encode(input_str))
        total_tokens_length = len(output[0].cpu())
        assert total_tokens_length > input_token_length, "Total tokens length must be greater than input token length"
        output_token_length = total_tokens_length - input_token_length
        assert output_token_length <= max_new_tokens, "Output token length must be less than or equal to max_new_tokens"

        decoded_answer = self.model.tokenizer.decode(output[0][input_token_length :].cpu())

        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(
                content=decoded_answer,
                model=self.model_name,
                source="generate",
            ),
        )

        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=input_token_length,
                output_tokens=output_token_length,
                total_tokens=total_tokens_length,
            ),
        )


def default_chat_history_parser(messages: list[ChatMessage]) -> str:
    """Convert list of content in `ChatMessageAssistant`, `ChatMessageUser` or `ChatMessageSystem` to a string.

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
                    "NNSight provider does not support multimodal content, please provide text inputs only."
                )
            message.content = message.text
        out += f"{message.role}: {message.content}\n"

    return out


def default_hook(input: str) -> None:
    """Default hook that does nothing."""
    pass