from openai import AsyncOpenAI
from typing import List, Dict, Any, Union, AsyncIterable
import json

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        stream: bool = False,
        json_mode: bool = False
    ) -> Union[str, AsyncIterable[str]]:
        try:
            # Only use response_format for models that support it
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            # Add response_format only for models that support it
            if json_mode and model in ["gpt-4", "gpt-4-turbo-preview"]:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)

            if stream:
                async def generate():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return generate()
            else:
                content = response.choices[0].message.content
                # If json_mode is True but we couldn't use response_format,
                # try to parse the response as JSON
                if json_mode and "response_format" not in kwargs:
                    try:
                        return json.dumps(json.loads(content))
                    except json.JSONDecodeError:
                        return content
                return content

        except Exception as e:
            print(f"Error in OpenAI completion: {str(e)}")
            raise 