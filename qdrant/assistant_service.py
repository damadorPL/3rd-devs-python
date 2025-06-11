from typing import Dict, Any
from openai_service import OpenAIService
from langfuse_service import LangfuseService

class AssistantService:
    def __init__(self, openai_service: OpenAIService, langfuse_service: LangfuseService):
        self.openai_service = openai_service
        self.langfuse_service = langfuse_service
    
    def answer(self, config: Dict[str, Any], trace) -> Dict[str, Any]:
        messages = config.get("messages", [])
        context = config.get("context", "")
        model = config.get("model", "gpt-4o")
        stream = config.get("stream", False)
        json_mode = config.get("json_mode", False)
        max_tokens = config.get("max_tokens", 4096)
        
        # Get prompt from Langfuse
        prompt = self.langfuse_service.get_prompt('Answer', version=1)
        compiled_prompt = prompt.compile(context=context)
        if isinstance(compiled_prompt, list) and len(compiled_prompt) > 0:
            system_message = compiled_prompt[0]
        else:
            system_message = {
                "role": "system",
                "content": compiled_prompt if isinstance(compiled_prompt, str) else str(compiled_prompt)
            }
            
        # Build thread
        thread = [system_message] + [msg for msg in messages if msg.get("role") != "system"]
        
        # Create generation in Langfuse
        generation = self.langfuse_service.create_generation(
            trace, 
            "Answer", 
            thread, 
            prompt, 
            {"model": model, "max_tokens": max_tokens}
        )
        
        try:
            completion = self.openai_service.completion({
                "messages": thread,
                "model": model,
                "stream": stream,
                "json_mode": json_mode,
                "max_tokens": max_tokens
            })
            
            self.langfuse_service.finalize_generation(
                generation,
                completion["choices"][0]["message"],
                completion["model"],
                {
                    "prompt_tokens": completion["usage"]["prompt_tokens"],
                    "completion_tokens": completion["usage"]["completion_tokens"],
                    "total_tokens": completion["usage"]["total_tokens"]
                }
            )
            
            return completion
            
        except Exception as error:
            self.langfuse_service.finalize_generation(
                generation,
                {"error": str(error)},
                "unknown"
            )
            raise error
