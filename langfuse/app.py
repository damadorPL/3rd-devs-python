from flask import Flask, request, jsonify, Response
import uuid
from typing import Dict, List, Any
from chat_service import ChatService
from langfuse_service import LangfuseService
from middleware.error_handler import error_handler

app = Flask(__name__)
port = 3000

chat_service = ChatService()
langfuse_service = LangfuseService()

@app.route('/api/chat', methods=['POST'])
def chat() -> Response:
    data: Dict[str, Any] = request.get_json()
    messages: List[Dict[str, str]] = data.get('messages', [])
    conversation_id: str = data.get('conversation_id', str(uuid.uuid4()))
    
    trace = langfuse_service.create_trace({
        'id': str(uuid.uuid4()),
        'name': "Chat",
        'session_id': conversation_id
    })

    try:
        all_messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant.", "name": "Alice"},
            *messages
        ]
        
        generated_messages: List[Dict[str, str]] = []

        # Main Completion - Answer user's question
        main_span = langfuse_service.create_span(trace, "Main Completion", all_messages)
        main_completion = chat_service.completion(all_messages, "gpt-4o")
        langfuse_service.finalize_span(main_span, "Main Completion", all_messages, main_completion)
        main_message = main_completion.choices[0].message
        all_messages.append(main_message)
        generated_messages.append(main_message)

        # Secondary Completion - Custom message
        secondary_messages: List[Dict[str, str]] = [{"role": "user", "content": "Please say 'completion 2'"}]
        secondary_span = langfuse_service.create_span(trace, "Secondary Completion", secondary_messages)
        secondary_completion = chat_service.completion(secondary_messages, "gpt-4o")
        langfuse_service.finalize_span(secondary_span, "Secondary Completion", secondary_messages, secondary_completion)
        secondary_message = secondary_completion.choices[0].message
        generated_messages.append(secondary_message)

        # Third Completion - Another custom message
        third_messages: List[Dict[str, str]] = [{"role": "user", "content": "Please say 'completion 3'"}]
        third_span = langfuse_service.create_span(trace, "Third Completion", third_messages)
        third_completion = chat_service.completion(third_messages, "gpt-4o")
        langfuse_service.finalize_span(third_span, "Third Completion", third_messages, third_completion)
        third_message = third_completion.choices[0].message
        generated_messages.append(third_message)

        # Finalize trace
        langfuse_service.finalize_trace(trace, messages, generated_messages)

        return jsonify({
            'completion': main_completion.choices[0].message.content,
            'completion2': secondary_completion.choices[0].message.content,
            'completion3': third_completion.choices[0].message.content,
            'conversation_id': conversation_id
        })
    except Exception as e:
        return error_handler(e)

@app.errorhandler(Exception)
def handle_error(error: Exception) -> Response:
    return error_handler(error)

if __name__ == '__main__':
    app.run(port=port) 