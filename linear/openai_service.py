import json
from typing import List, Dict, Any, Union, AsyncIterator
from openai import OpenAI
from prompts import projectAssignmentPrompt

class ProjectAssignment:
    def __init__(self, thoughts: str, name: str, id: str):
        self._thoughts = thoughts
        self.name = name
        self.id = id

class OpenAIService:
    def __init__(self):
        self.openai = OpenAI()
    
    async def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        stream: bool = False,
        json_mode: bool = False
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Handles OpenAI API interactions for chat completions and embeddings.
        Uses OpenAI's chat.completions and embeddings APIs.
        Supports streaming, JSON mode, and different models.
        Logs interactions to a prompt.md file for debugging.
        """
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                response_format=response_format
            )
            
            if stream:
                return chat_completion
            else:
                return chat_completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise error
    
    async def createEmbedding(self, input_text: Union[str, List[str]]) -> List[float]:
        """
        Creates an embedding for the given input using OpenAI's text-embedding-3-large model.
        
        Args:
            input_text: A string or array of strings to create embeddings for.
        
        Returns:
            A list of numbers representing the embedding.
        
        Raises:
            Exception: If there's an issue creating the embedding.
        """
        print(input_text)
        try:
            embedding = await self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=input_text,
                encoding_format="float"
            )
            
            # Return the embedding vector
            return embedding.data[0].embedding
        except Exception as error:
            print("Error in creating embedding:", error)
            raise error
    
    async def assignProjectToTask(self, title: str, description: str) -> ProjectAssignment:
        prompt = f"""Please assign this task to the project:

Title: {title}
Description: {description}"""
        
        messages = [
            {"role": "system", "content": projectAssignmentPrompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = await self.completion(messages, "gpt-4o", False, True)
            
            if hasattr(completion, 'choices') and completion.choices[0].message.content:
                result_dict = json.loads(completion.choices[0].message.content)
                return ProjectAssignment(
                    thoughts=result_dict["_thoughts"],
                    name=result_dict["name"],
                    id=result_dict["id"]
                )
            else:
                raise Exception("Unexpected response format from OpenAI")
        except Exception as error:
            print("Error in assigning project to task:", error)
            raise error
