import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
import tiktoken

class OpenAIService:
    """Service class for OpenAI API interactions with token counting and image processing"""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tokenizers: Dict[str, tiktoken.Encoding] = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    def _get_tokenizer(self, model_name: str) -> tiktoken.Encoding:
        """Get or create tokenizer for the specified model"""
        if model_name not in self.tokenizers:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
                self.tokenizers[model_name] = encoding
            except KeyError:
                # Fallback to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
                self.tokenizers[model_name] = encoding
        
        return self.tokenizers[model_name]
    
    async def count_tokens(self, messages: List[Dict[str, Any]], model: str = 'gpt-4o') -> int:
        """Count tokens in messages using tiktoken"""
        tokenizer = self._get_tokenizer(model)
        
        formatted_content = ''
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle multimodal content
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                content = ' '.join(text_parts)
            
            formatted_content += f"{self.IM_START}{message['role']}{self.IM_SEP}{content}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        
        tokens = tokenizer.encode(formatted_content)
        return len(tokens)
    
    async def completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4",
        stream: bool = False,
        json_mode: bool = False,
        max_tokens: int = 1024
    ) -> Any:
        """Create chat completion with OpenAI API"""
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            return chat_completion
                
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error
    
    async def calculate_image_tokens(self, width: int, height: int, detail: str) -> int:
        """Calculate token cost for image processing"""
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
        import math
        num_squares = math.ceil(width / 512) * math.ceil(height / 512)
        
        # Calculate the token cost
        token_cost += (num_squares * 170) + 85
        return token_cost
