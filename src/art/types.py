from dataclasses import dataclass, field
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

Message = ChatCompletionMessageParam | Choice
Messages = list[Message]


@dataclass
class Trajectory:
    messages: Messages
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)
