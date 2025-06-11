import json
from typing import List, Dict, Any
from openai import OpenAI
import tiktoken

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()
        self.tokenizers = {}
        
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    def get_tokenizer(self, model_name: str = "gpt-4o"):
        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = tiktoken.encoding_for_model(model_name)
        return self.tokenizers[model_name]
    
    def count_tokens(self, messages: List[Dict[str, Any]], model: str = "gpt-4o") -> int:
        tokenizer = self.get_tokenizer(model)
        formatted_content = ""
        
        for message in messages:
            formatted_content += f"{self.IM_START}{message['role']}{self.IM_SEP}{message.get('content', '')}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        
        tokens = tokenizer.encode(formatted_content)
        return len(tokens)
    
    def completion(self, config: Dict[str, Any]) -> Dict[str, Any]:
        messages = config.get("messages", [])
        model = config.get("model", "gpt-4o")
        stream = config.get("stream", False)
        json_mode = config.get("json_mode", False)
        max_tokens = config.get("max_tokens", 4096)
        temperature = config.get("temperature", 0)
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object" if json_mode else "text"}
            )
            
            if stream:
                return response
            else:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": response.choices[0].message.content,
                                "role": response.choices[0].message.role
                            }
                        }
                    ],
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
        
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error
    
    def parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                raise ValueError("Invalid response structure")
            
            return json.loads(content)
        
        except Exception as error:
            print(f"Error parsing JSON response: {error}")
            return {"error": "Failed to process response", "result": False}
    
    def create_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        
        except Exception as error:
            print(f"Error creating embedding: {error}")
            raise error
