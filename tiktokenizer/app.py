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
    
    try:
        formatted_content, token_count = openai_service.count_tokens(messages, model)
        print(f"Formatted content: {formatted_content}")
        print(f"Token count for model {model}: {token_count}")
        return jsonify({"tokenCount": token_count, "model": model})
    except Exception as error:
        print(f"Error in token counting: {error}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    print(f"Server running at http://localhost:{port}. Listening for POST /api/chat requests")