import uuid
from flask import Flask, request, jsonify
from langfuse_service import LangfuseService
from openai_service import OpenAIService
from assistant_service import AssistantService
from vector_service import VectorService

app = Flask(__name__)

def validate_chat_request(request_data):
    if not isinstance(request_data, dict):
        return False, "Request must be a JSON object"
    
    if "messages" not in request_data:
        return False, "Request must contain 'messages' field"
    
    if not isinstance(request_data["messages"], list):
        return False, "Messages must be a list"
    
    for msg in request_data["messages"]:
        if not isinstance(msg, dict):
            return False, "Each message must be an object"
        if "role" not in msg or "content" not in msg:
            return False, "Each message must have 'role' and 'content' fields"
    
    return True, None

# Initialize services
langfuse_service = LangfuseService()
openai_service = OpenAIService()
assistant_service = AssistantService(openai_service, langfuse_service)
vector_service = VectorService(openai_service)

COLLECTION_NAME = "aidevs"

@app.route("/api/chat", methods=["POST"])
def chat():
    request_data = request.json
    valid, error_message = validate_chat_request(request_data)
    
    if not valid:
        return jsonify({"error": error_message}), 400
    
    messages = [msg for msg in request_data["messages"] if msg["role"] != 'system']
    conversation_id = request_data.get("conversation_id") or str(uuid.uuid4())
    
    trace = langfuse_service.create_trace({
        "id": str(uuid.uuid4()),
        "name": (messages[-1]["content"] or "")[:45],
        "session_id": conversation_id,
        "user_id": "test-user"
    })
    
    try:
        vector_service.ensure_collection(COLLECTION_NAME)
        
        # Search for similar messages
        last_message = messages[-1]
        similar_messages = vector_service.perform_search(
            COLLECTION_NAME, 
            last_message["content"], 
            10
        )
        
        # Add similar messages to context
        context_parts = []
        for result in similar_messages:
            role = "Adam" if result.payload.get("role") == "user" else "Alice"
            context_parts.append(f"{role}: {result.payload.get('text')}")
        context = '\n'.join(context_parts)
        
        # Generate answer
        answer = assistant_service.answer({
            "messages": messages,
            "context": context
        }, trace)
        
        # Save new messages to vector store
        vector_service.add_points(COLLECTION_NAME, [
            {
                "id": str(uuid.uuid4()),
                "text": last_message["content"],
                "role": last_message["role"]
            },
            {
                "id": str(uuid.uuid4()),
                "text": answer["choices"][0]["message"]["content"],
                "role": "assistant"
            }
        ])
        
        langfuse_service.finalize_trace(
            trace, 
            request_data, 
            answer["choices"][0]["message"]
        )
        langfuse_service.flush()
        
        response_data = {**answer, "conversation_id": conversation_id}
        return jsonify(response_data)
        
    except Exception as error:
        langfuse_service.finalize_trace(
            trace, 
            request_data, 
            {"error": "An error occurred while processing your request"}
        )
        print(f"Error in chat processing: {error}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=3000, debug=True)
