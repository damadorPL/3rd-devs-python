from flask import Flask, request, jsonify
from typing import TypedDict, List, Optional, Literal
from OpenAIService import OpenAIService
from WebSearch import WebSearchService
from prompts import answer_prompt
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

Role = Literal['user', 'assistant', 'system']

class Message(TypedDict):
    role: Role
    content: str
    name: Optional[str]

class SearchResult(TypedDict):
    url: str
    title: str
    description: str
    content: Optional[str]

allowed_domains = [
    {'name': 'Wikipedia', 'url': 'en.wikipedia.org', 'scrappable': True},
    {'name': 'easycart', 'url': 'easycart.pl', 'scrappable': True},
    {'name': 'FS.blog', 'url': 'fs.blog', 'scrappable': True},
    {'name': 'arXiv', 'url': 'arxiv.org', 'scrappable': True},
    {'name': 'Instagram', 'url': 'instagram.com', 'scrappable': False},
    {'name': 'OpenAI', 'url': 'openai.com', 'scrappable': True},
    {'name': 'Brain overment', 'url': 'brain.overment.com', 'scrappable': True},
]

app = Flask(__name__)
port = 3000

web_search_service = WebSearchService(
    allowed_domains=allowed_domains,
    api_key=os.getenv('FIRECRAWL_API_KEY', '')
)
openai_service = OpenAIService()

@app.route('/api/chat', methods=['POST'])
async def chat():
    print('Received request')
    with open('prompt.md', 'w') as f:
        f.write('')
    
    data = request.get_json()
    messages: List[Message] = data.get('messages', [])

    try:
        latest_user_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
        if not latest_user_message:
            raise ValueError('No user message found')

        print(f"Processing user message: {latest_user_message['content']}")
        
        should_search = await web_search_service.is_web_search_needed(latest_user_message['content'])
        print(f"Should search: {should_search}")
        
        merged_results: List[SearchResult] = []

        if should_search:
            print("Starting web search process...")
            queries_result = await web_search_service.generate_queries(latest_user_message['content'])
            print(f"Generated queries: {queries_result}")
            
            if queries_result['queries']:
                search_results = await web_search_service.search_web(queries_result['queries'])
                print(f"Search results: {search_results}")
                
                filtered_results = await web_search_service.score_results(search_results, latest_user_message['content'])
                print(f"Filtered results: {filtered_results}")
                
                urls_to_load = await web_search_service.select_resources_to_load(latest_user_message['content'], filtered_results)
                print(f"URLs to load: {urls_to_load}")
                
                scraped_content = await web_search_service.scrape_urls(urls_to_load)
                print(f"Scraped content: {scraped_content}")
                
                merged_results = []
                for result in filtered_results:
                    scraped_item = next((item for item in scraped_content if item['url'] == result['url']), None)
                    if scraped_item:
                        merged_results.append({**result, 'content': scraped_item['content']})
                    else:
                        merged_results.append(result)
                print(f"Merged results: {merged_results}")

        prompt_with_results = answer_prompt(merged_results)
        all_messages = [
            {'role': 'system', 'content': prompt_with_results, 'name': 'Alice'},
            *messages
        ]
        completion = await openai_service.completion(all_messages, "gpt-4o", False)

        return jsonify(completion)
    except Exception as error:
        print('Error in chat processing:', error)
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/api/chat-dummy', methods=['POST'])
async def chat_dummy():
    print('Received dummy request')
    data = request.get_json()
    messages: List[Message] = data.get('messages', [])

    try:
        latest_user_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
        if not latest_user_message:
            raise ValueError('No user message found')

        dummy_response = {
            'role': 'assistant',
            'content': f'This is a dummy response to: "{latest_user_message["content"]}"'
        }

        completion = {
            'id': 'dummy-completion-id',
            'object': 'chat.completion',
            'created': int(time.time() * 1000),
            'model': 'dummy-model',
            'choices': [
                {
                    'index': 0,
                    'message': dummy_response,
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }

        return jsonify(completion)
    except Exception as error:
        print('Error in chat processing:', error)
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(port=port, debug=True)
    print(f'Server running at http://localhost:{port}. Listening for POST /api/chat requests') 