import os
import base64
from typing import List, Dict, Any, Union, Optional, AsyncIterable
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
import aiohttp
import asyncio
from text_service import TextService, IDoc

@dataclass
class ImageProcessingResult:
    description: str
    source: str

class OpenAIService:
    def __init__(self):
        self.openai = AsyncOpenAI()
        self.text_service = TextService()

    async def completion(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o",
        stream: bool = False,
        json_mode: bool = False,
        max_tokens: int = 8096,
        config: Optional[Dict[str, Any]] = None
    ) -> Union[ChatCompletion, AsyncIterable[ChatCompletionChunk]]:
        # Handle both old and new parameter styles
        if config is not None:
            messages = config.get("messages", [])
            model = config.get("model", model)
            stream = config.get("stream", stream)
            json_mode = config.get("jsonMode", json_mode)
            max_tokens = config.get("maxTokens", max_tokens)

        if messages is None:
            raise ValueError("Messages must be provided either directly or through config")

        try:
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream if model not in ['o1-mini', 'o1-preview'] else False,
                max_tokens=max_tokens if model not in ['o1-mini', 'o1-preview'] else None,
                response_format={"type": "json_object"} if json_mode else {"type": "text"}
            )

            return chat_completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise error

    async def process_image(self, image_path: str) -> ImageProcessingResult:
        try:
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = await self.openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )

            return ImageProcessingResult(
                description=response.choices[0].message.content or "No description available.",
                source=image_path
            )
        except Exception as error:
            print(f"Error processing image {image_path}:", error)
            raise error

    async def process_images(self, image_paths: List[str]) -> List[ImageProcessingResult]:
        try:
            tasks = [self.process_image(path) for path in image_paths]
            results = await asyncio.gather(*tasks)
            return results
        except Exception as error:
            print("Error processing multiple images:", error)
            raise error

    async def transcribe_buffer(
        self,
        audio_buffer: bytes,
        config: Dict[str, str] = {"language": "en", "prompt": ""}
    ) -> str:
        print("Transcribing audio...")

        try:
            # Create a temporary file-like object from the buffer
            from io import BytesIO
            audio_file = BytesIO(audio_buffer)

            transcription = await self.openai.audio.transcriptions.create(
                file=audio_file,
                language=config["language"],
                model="whisper-1",
                prompt=config.get("prompt", "")
            )
            return transcription.text
        except Exception as error:
            print("Error transcribing audio:", error)
            raise error

    async def transcribe(
        self,
        audio_files: List[str],
        config: Dict[str, str] = {"language": "pl", "prompt": "", "fileName": "transcription.md"}
    ) -> List[IDoc]:
        print("Transcribing multiple audio files...")

        async def process_file(file_path: str) -> IDoc:
            with open(file_path, "rb") as f:
                buffer = f.read()
            transcription = await self.transcribe_buffer(
                buffer,
                {"language": config["language"], "prompt": config.get("prompt", "")}
            )
            doc = await self.text_service.document(
                transcription,
                "gpt-4o",
                {"source": file_path, "name": config["fileName"]}
            )
            return doc

        try:
            tasks = [process_file(file_path) for file_path in audio_files]
            results = await asyncio.gather(*tasks)
            return results
        except Exception as error:
            print("Error transcribing multiple files:", error)
            raise error

    async def create_embedding(self, text: str) -> List[float]:
        try:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as error:
            print("Error creating embedding:", error)
            raise error

    async def create_jina_embedding(self, text: str) -> List[float]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.jina.ai/v1/embeddings',
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
                    },
                    json={
                        'model': 'jina-embeddings-v3',
                        'task': 'text-matching',
                        'dimensions': 1024,
                        'late_chunking': False,
                        'embedding_type': 'float',
                        'input': [text]
                    }
                ) as response:
                    if not response.ok:
                        raise Exception(f'HTTP error! status: {response.status}')
                    data = await response.json()
                    return data['data'][0]['embedding']
        except Exception as error:
            print("Error creating Jina embedding:", error)
            raise error
