from typing import Dict, List, Optional, Union, AsyncIterator
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import tiktoken
import math

class OpenAIService:
    """OpenAI service wrapper with token counting capabilities and image processing support."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client instance."""
        self.openai = openai.AsyncOpenAI(api_key=api_key)
        self.tokenizers: Dict[str, tiktoken.Encoding] = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    async def get_tokenizer(self, model_name: str) -> tiktoken.Encoding:
        """Get or create a tokenizer for the specified model"""
        if model_name not in self.tokenizers:
            try:
                self.tokenizers[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                raise KeyError(f"Tokenizer for model {model_name} not found.")
        return self.tokenizers[model_name]
    
    async def count_tokens(
        self, 
        messages: List[ChatCompletionMessageParam], 
        model: str = 'gpt-4o'
    ) -> int:
        """Count the number of tokens in the given messages for the specified model"""
        tokenizer = await self.get_tokenizer(model)
        
        formatted_content = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "") or ""
            formatted_content += f"{self.IM_START}{role}{self.IM_SEP}{content}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        tokens = tokenizer.encode(formatted_content)
        return len(tokens)
    
    async def completion(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str = "gpt-4o",
        stream: bool = False,
        json_mode: bool = False,
        max_tokens: int = 4096
    ) -> Union[ChatCompletion, AsyncIterator]:
        """Create a completion with the OpenAI API"""
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            # Use appropriate token parameter based on model
            if model.startswith('o1-'):
                return await self.openai.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    max_completion_tokens=max_tokens,
                    response_format=response_format
                )
            else:
                return await self.openai.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error
    
    async def calculate_image_tokens(
        self, 
        width: int, 
        height: int, 
        detail: str = 'high'
    ) -> int:
        """Calculate the token cost for processing an image based on its dimensions and detail level."""
        token_cost = 0
        
        if detail == 'low':
            token_cost += 85
            return token_cost
        
        MAX_DIMENSION = 2048
        SCALE_SIZE = 768
        
        # Resize to fit within MAX_DIMENSION x MAX_DIMENSION
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                width = MAX_DIMENSION
                height = round(MAX_DIMENSION / aspect_ratio)
            else:
                height = MAX_DIMENSION
                width = round(MAX_DIMENSION * aspect_ratio)
        
        # Scale the shortest side to SCALE_SIZE
        if width >= height and height > SCALE_SIZE:
            width = round((SCALE_SIZE / height) * width)
            height = SCALE_SIZE
        elif height > width and width > SCALE_SIZE:
            height = round((SCALE_SIZE / width) * height)
            width = SCALE_SIZE
        
        # Calculate the number of 512px squares
        num_squares = math.ceil(width / 512) * math.ceil(height / 512)
        
        # Calculate the token cost
        token_cost += (num_squares * 170) + 85
        return token_cost
