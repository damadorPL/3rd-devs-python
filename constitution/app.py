from flask import Flask, request, jsonify
import json
import os
from openai_service import OpenAIService
from prompts import verification_prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Start Flask server
app = Flask(__name__)
port = 3000

openai_service = OpenAIService(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    model = data.get('model', 'gpt-4o')
    
    if not messages or not messages[-1].get('content'):
        return jsonify({"error": 'Valid message content is required'}), 400
    
    last_message = messages[-1]
    
    try:
        last_message_content = last_message['content']
        if not isinstance(last_message_content, str):
            last_message_content = json.dumps(last_message_content)
            
        verification_response = openai_service.completion(
            messages=[
                {"role": "system", "content": verification_prompt},
                {"role": "user", "content": last_message_content}
            ],
            model=model,
            stream=False
        )
        
        if verification_response.choices[0].message.content != 'pass':
            return jsonify({"error": 'Message is not in Polish'}), 400
        
        full_response = openai_service.completion(
            messages=messages,
            model=model
        )
        
        return jsonify({
            "role": "assistant",
            "content": full_response.choices[0].message.content
        })
        
    except Exception as error:
        print(f'Error: {str(error)}')
        return jsonify({"error": 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    print(f'Server running at http://localhost:{port}. Listening for POST /api/chat requests')
    app.run(host='0.0.0.0', port=port)
