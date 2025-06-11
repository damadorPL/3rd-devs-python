from openai import OpenAI
from typing import List, Dict, Any
import tiktoken

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()
        self.tokenizers: Dict[str, tiktoken.Encoding] = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    def _get_tokenizer(self, model_name: str) -> tiktoken.Encoding:
        if model_name not in self.tokenizers:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            self.tokenizers[model_name] = encoding
        return self.tokenizers[model_name]
    
    def count_tokens(self, messages: List[Dict[str, Any]], model: str = 'gpt-4o') -> int:
        tokenizer = self._get_tokenizer(model)
        formatted_content = ''
        
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle multimodal content
                text_content = ''
                for item in content:
                    if item.get('type') == 'text':
                        text_content += item.get('text', '')
                content = text_content
            
            formatted_content += f"{self.IM_START}{message['role']}{self.IM_SEP}{content}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        tokens = tokenizer.encode(formatted_content)
        return len(tokens)
    
    def completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
        stream: bool = False,
        json_mode: bool = False,
        max_tokens: int = 4096
    ):
        try:
            kwargs = {
                "messages": messages,
                "model": model
            }
            
            # Don't set these parameters for o1 models
            if model not in ['o1-mini', 'o1-preview']:
                kwargs.update({
                    "stream": stream,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"} if json_mode else {"type": "text"}
                })
            
            chat_completion = self.client.chat.completions.create(**kwargs)
            return chat_completion
            
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error
    
    def calculate_image_tokens(self, width: int, height: int, detail: str = 'high') -> int:
        token_cost = 0
        
        if detail == 'low':
            return 85
        
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
