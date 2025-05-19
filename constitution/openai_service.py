import openai
from typing import List, Dict, Any
import tiktoken

class OpenAIService:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.openai = openai.OpenAI(api_key=self.api_key)
        self.tokenizers = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    def get_tokenizer(self, model_name: str):
        """Get or create a tokenizer for the specified model"""
        if model_name not in self.tokenizers:
            try:
                self.tokenizers[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                KeyError(f"Tokenizer for model {model_name} not found.")
        return self.tokenizers[model_name]
    
    async def count_tokens(self, messages: List[Dict[str, str]], model: str = 'gpt-4o') -> int:
        tokenizer = self.get_tokenizer(model)
        formatted_content = ''
        
        for message in messages:
            formatted_content += f"{self.IM_START}{message.get('role')}{self.IM_SEP}{message.get('content', '')}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        
        # Use tiktoken to count tokens
        tokens = tokenizer.encode(formatted_content)
        return len(tokens)
    
    def completion(self, messages: List[Dict[str, Any]], model: str = "gpt-4", 
                  stream: bool = False, json_mode: bool = False, max_tokens: int = 1024):
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            chat_completion = self.openai.chat.completions.create(
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
