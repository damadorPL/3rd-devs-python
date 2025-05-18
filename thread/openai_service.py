import openai

class OpenAIService:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def completion(self, messages, model="gpt-4", stream=False):
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream
            )
            
            return response
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error