from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from openai_service import OpenAIService

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
port = 3000

# Initialize OpenAI service
openai_service = OpenAIService(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    model = data.get('model', 'gpt-4o')
    
    if not messages:
        return jsonify({"error": "Messages are required"}), 400
    
    try:
        model_context_length = 128000
        max_output_tokens = 50
        input_tokens = openai_service.count_tokens(messages, model)
        
        if input_tokens + max_output_tokens > model_context_length:
            return jsonify({
                "error": f"No space left for response. Input tokens: {input_tokens}, Context length: {model_context_length}"
            }), 400
        
        print(f"Input tokens: {input_tokens}, Max tokens: {max_output_tokens}, " +
              f"Model context length: {model_context_length}, " +
              f"Tokens left: {model_context_length - (input_tokens + max_output_tokens)}")
        
        full_response = openai_service.continuous_completion(
            messages=messages,
            model=model,
            max_tokens=max_output_tokens
        )
        
        return jsonify({
            "role": "assistant",
            "content": full_response
        })
        
    except Exception as error:
        print(f"Error: {error}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"Server running at http://localhost:{port}. Listening for POST /api/chat requests")
