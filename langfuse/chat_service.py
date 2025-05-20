from typing import List, Dict
from openai_service import OpenAIService
from openai.types.chat import ChatCompletion

class ChatService:
    def __init__(self) -> None:
        self.openai_service = OpenAIService()

    def completion(self, messages: List[Dict[str, str]], model: str) -> ChatCompletion:
        return self.openai_service.completion({
            'messages': messages,
            'model': model,
            'stream': False,
            'json_mode': False
        }) 