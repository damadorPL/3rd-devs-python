import tiktoken
import openai

class OpenAIService:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.tokenizers = {}
        self.IM_START = "<|im_start|>"
        self.IM_END = "<|im_end|>"
        self.IM_SEP = "<|im_sep|>"
    
    def get_tokenizer(self, model_name):
        """Get or create a tokenizer for the specified model"""
        if model_name not in self.tokenizers:
            try:
                self.tokenizers[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                raise KeyError(f"Tokenizer for model {model_name} not found.")
        return self.tokenizers[model_name]

    def count_tokens(self, messages, model="gpt-4o"):
        """Count the number of tokens in the given messages for the specified model"""
        tokenizer = self.get_tokenizer(model)
        
        formatted_content = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "") or ""
            formatted_content += f"{self.IM_START}{role}{self.IM_SEP}{content}{self.IM_END}"
        
        formatted_content += f"{self.IM_START}assistant{self.IM_SEP}"
        
        # Encode the formatted content to count tokens
        tokens = tokenizer.encode(formatted_content)
        
        return len(tokens)
    
    def completion(self, messages, model="gpt-4", stream=False, json_mode=False, max_tokens=1024):
        """Create a completion with the OpenAI API"""
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            return self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                max_tokens=max_tokens,
                response_format=response_format
            )
        except Exception as error:
            print(f"Error in OpenAI completion: {error}")
            raise error
    
    def continuous_completion(self, messages, model="gpt-4o", max_tokens=1024):
        """Create a completion that continues if the response is cut off"""
        full_response = ""
        is_completed = False
        
        while not is_completed:
            completion = self.completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens
            )
            
            choice = completion.choices[0]
            content = choice.message.content or ""
            full_response += content
            
            if choice.finish_reason != "length":
                is_completed = True
            else:
                print("Continuing completion...")
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "[system: Please continue your response to the user's question and finish when you're done from the very next character you were about to write, because you didn't finish your response last time. At the end, your response will be concatenated with the last completion.]"}
                ]
                
        return full_response
