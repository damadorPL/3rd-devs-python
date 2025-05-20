from typing import Dict, List, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIService:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def completion(self, config: Dict[str, Any]) -> ChatCompletion:
        messages: List[Dict[str, str]] = config.get('messages', [])
        model: str = config.get('model', 'gpt-4')
        stream: bool = config.get('stream', False)
        json_mode: bool = config.get('json_mode', False)

        response = self.openai.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            response_format={"type": "json_object"} if json_mode else {"type": "text"}
        )
        return response 