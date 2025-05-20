from typing import Dict, List, Optional
from langfuse import Langfuse
from openai.types.chat import ChatCompletion
import os
from dotenv import load_dotenv
import json

load_dotenv()

class LangfuseService:
    def __init__(self) -> None:
        self.langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST')
        )

        if os.getenv('FLASK_ENV') == 'development' or os.getenv('FLASK_DEBUG') == '1':
            self.langfuse.debug()

    def create_trace(self, options: Dict[str, str]):
        return self.langfuse.trace(**options)

    def create_span(self, trace, name: str, input_data: Optional[List[Dict[str, str]]] = None):
        return trace.span(
            name=name,
            input=input_data if input_data else None
        )

    def finalize_span(self, span, name: str, input_data: List[Dict[str, str]], output: ChatCompletion) -> None:
        span.update(
            name=name,
            output=output.choices[0].message
        )

        generation = span.generation(
            name=name,
            model=output.model,
            model_parameters={
                'temperature': 0.7
            },
            input=input_data,
            output=output,
            usage={
                'prompt_tokens': output.usage.prompt_tokens,
                'completion_tokens': output.usage.completion_tokens,
                'total_tokens': output.usage.total_tokens
            }
        )
        generation.end()
        span.end()

    def finalize_trace(self, trace, original_messages: List[Dict[str, str]], generated_messages: List[Dict[str, str]]) -> None:
        input_messages = [msg for msg in original_messages if msg['role'] != 'system']
        trace.update(
            input=input_messages,
            output=generated_messages
        )
        self.langfuse.flush()

    def shutdown(self) -> None:
        self.langfuse.shutdown() 