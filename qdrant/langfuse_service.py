import os
from typing import Dict, Any, List
from langfuse import Langfuse

class LangfuseService:
    def __init__(self):
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        
        if os.getenv("NODE_ENV") == "development":
            self.langfuse.debug()
    
    def flush(self):
        "Forces immediate sending of any pending telemetry data to Langfuse servers"
        return self.langfuse.flush()
    
    def create_trace(self, options: Dict[str, str]):
        "Creates a new top-level trace with specified ID, name, session, and user identifiers"
        return self.langfuse.trace(
            id=options["id"],
            name=options["name"],
            session_id=options["session_id"],
            user_id=options["user_id"]
        )
    
    def create_span(self, trace, name: str, input_data: Any = None):
        "Creates a child span within a trace to track sub-operations with optional input data"
        return trace.span(name=name, input=input_data if input_data else None)
    
    def finalize_span(self, span, name: str, output: Any):
        "Updates the span with final output data and marks it as completed"
        span.update(name=name, output=output)
        span.end()
    
    def finalize_trace(self, trace, input_data: Any, output: Any):
        "Updates the trace with final input/output data and flushes all pending information"
        trace.update(input=input_data, output=output)
        self.flush()
    
    def shutdown(self):
        "Cleanly terminates the Langfuse client connection"
        self.langfuse.shutdown()
    
    def create_generation(self, trace, name: str, input_data: Any, prompt=None, config: Dict[str, Any] = None):
        "Creates a generation object to track LLM model interactions within a trace"
        return trace.generation(
            name=name,
            input=input_data,
            prompt=prompt,
            **(config or {})
        )
    
    def create_event(self, trace, name: str, input_data: Any = None, output: Any = None):
        "Logs a discrete event with optional input/output data to a trace for debugging purposes"
        trace.event(
            name=name,
            input=str(input_data) if input_data else None,
            output=str(output) if output else None
        )
    
    def finalize_generation(self, generation, output: Any, model: str, usage: Dict[str, int] = None):
        "Updates the generation with output, model name, and usage statistics then marks it complete"
        generation.update(
            output=output,
            model=model,
            usage=usage
        )
        generation.end()
    
    def get_prompt(self, name: str, version: int = None, **kwargs):
        "Retrieves a prompt template by name with optional version specification from Langfuse"
        if version is not None:
            return self.langfuse.get_prompt(name, version=version, **kwargs)
        else:
            return self.langfuse.get_prompt(name, **kwargs)
    
    def pre_fetch_prompts(self, prompt_names: List[str]):
        "Loads multiple prompt templates into cache by iteratively fetching them"
        for name in prompt_names:
            self.get_prompt(name)
