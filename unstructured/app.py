import os
import json
import asyncio
from pathlib import Path
from typing import List

from text_service import TextSplitter, IDoc
from openai_service import OpenAIService

# Initialize services
splitter = TextSplitter()
openai_service = OpenAIService()

# Constants
SOURCE_FILE = 'source.md'
OUTPUT_FILE = 'tools.json'
MAX_CHUNK_SIZE = 500

async def main():
    """Main function to orchestrate the process"""
    try:
        source_content = await load_source_file(os.path.dirname(__file__))
        extracted_tools = await extract_tools(source_content)
        split_docs = await split_content(extracted_tools)
        await save_output(split_docs)
        print('Process completed successfully. Check tools.json for results.')
    except Exception as error:
        print(f'An error occurred: {error}')

async def split_content(content: str) -> List[IDoc]:
    """New splitContent function"""
    chunks = content.split('\n\n')
    docs = []
    for chunk in chunks:
        doc = await splitter.document(chunk)
        docs.append(doc)
    return docs

# ALTERNATIVE SPLITTING
# async def split_content(content: str) -> List[IDoc]:
#     return await splitter.split(content, MAX_CHUNK_SIZE)

async def load_source_file(dirname: str) -> str:
    """Load the source file"""
    file_path = Path(dirname) / SOURCE_FILE
    return file_path.read_text(encoding='utf-8')

async def extract_tools(file_content: str) -> str:
    """Extract tools information using OpenAI"""
    user_message = {
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': f"""{file_content}

Please extract all the information from the article's content related to tools, apps, or software, including links and descriptions in markdown format. Ensure the list items are unique. Always separate each tool with a double line break. Respond only with the concise content and nothing else."""
        }]
    }

    response = await openai_service.completion([user_message], 'o1-mini', stream=False)
    content = response.choices[0].message.content or ''
    return content

async def save_output(docs: List[IDoc]):
    """Save the output to a JSON file"""
    output_path = Path(__file__).parent / OUTPUT_FILE
    output_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding='utf-8')

# Run the main function
if __name__ == '__main__':
    asyncio.run(main())
