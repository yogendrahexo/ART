from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from typing import Literal

Message = ChatCompletionMessageParam  # | Choice
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]


class Trajectory(BaseModel):
    messages: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}


Verbosity = Literal[0, 1, 2]
