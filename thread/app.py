from flask import Flask, request, jsonify
from openai_service import OpenAIService
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

app = Flask(__name__)
port = 3000

openai_service = OpenAIService(api_key=os.getenv("OPENAI_API_KEY"))
previous_summarization = ""

# Function to generate summarization based on the current turn and previous summarization
def generate_summarization(user_message, assistant_response):
    summarization_prompt = {
        "role": "system",
        "content": f"""Please summarize the following conversation in a concise manner, incorporating the previous summary if available:
        {previous_summarization or "No previous summary"}

        User: {user_message['content']}
        Assistant: {assistant_response.content}
        """
    }
    
    response = openai_service.completion(
        messages=[summarization_prompt, {"role": "user", "content": "Please create/update our conversation summary."}],
        model="gpt-4o-mini",
        stream=False
    )
    
    return response.choices[0].message.content or "No conversation history"

# Function to create system prompt
def create_system_prompt(summarization):
    return {
        "role": "system",
        "content": f"""You are Alice, a helpful assistant who speaks using as few words as possible.

        {f'Here is a summary of the conversation so far:\n\n{summarization}\n\n' if summarization else ''}Let's chat!"""
    }

@app.route('/api/chat', methods=['POST'])
def chat():
    global previous_summarization
    data = request.json
    message = data.get('message')
    
    try:
        system_prompt = create_system_prompt(previous_summarization)
        assistant_response = openai_service.completion(
            messages=[system_prompt, message],
            model="gpt-4o",
            stream=False
        )
        
        # Generate new summarization
        previous_summarization = generate_summarization(message, assistant_response.choices[0].message)
        
        return jsonify({
            "content": assistant_response.choices[0].message.content,
            "role": "assistant"
        })

    except Exception as error:
        print(f'Error in OpenAI completion: {error}')
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route('/api/demo', methods=['POST'])
def demo():
    global previous_summarization
    demo_messages = [
        {"content": "Hi! I'm Adam", "role": "user"},
        {"content": "How are you?", "role": "user"},
        {"content": "Do you know my name?", "role": "user"}
    ]
    
    assistant_response = None
    responses = []
    
    for message in demo_messages:
        print('--- NEXT TURN ---')
        print(f'Adam: {message["content"]}')
        
        try:
            system_prompt = create_system_prompt(previous_summarization)
            assistant_response = openai_service.completion(
                messages=[system_prompt, message],
                model="gpt-4o",
                stream=False
            )
            
            print(f'Alice: {assistant_response.choices[0].message.content}')
            responses.append({
                "content": assistant_response.choices[0].message.content,
                "role": "assistant"
            })
            
            # Generate new summarization
            previous_summarization = generate_summarization(message, assistant_response.choices[0].message)

        except Exception as error:
            print(f'Error in OpenAI completion: {error}')
            return jsonify({"error": "An error occurred while processing your request"}), 500
    
    return jsonify({"responses": responses})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"Server running at http://localhost:{port}. Listening for POST /api/chat requests")