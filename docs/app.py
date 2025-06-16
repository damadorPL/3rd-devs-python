import os
import asyncio
from file_service import FileService
from text_service import TextService
from openai_service import OpenAIService
from vector_service import VectorService
from search_service import SearchService
from database_service import DatabaseService
from document_service import DocumentService

from dotenv import load_dotenv
load_dotenv()

async def main():
    file_service = FileService()
    text_service = TextService()
    openai_service = OpenAIService()
    vector_service = VectorService(openai_service)
    search_service = SearchService(
        os.getenv('ALGOLIA_APP_ID', ''),
        os.getenv('ALGOLIA_API_KEY', '')
    )
    database_service = DatabaseService('docs/database.db', search_service, vector_service)
    document_service = DocumentService(openai_service, database_service, text_service)

    # Process file from URL
    url = 'https://cloud.overment.com/S04E03-1732688101.md'
    docs_result = await file_service.process(url, 2500)
    docs = docs_result['docs']

    # Insert documents into the database
    for doc in docs:
        await database_service.insert_document(doc, True)

    # Translate documents
    translated_docs = await document_service.translate(docs, 'Polish', 'English')
    merged_translation = '\n'.join(
        text_service.restore_placeholders(doc).text.strip() for doc in translated_docs
    )

    # Save translation result to result.md
    result_path = os.path.join(os.path.dirname(__file__), 'result.md')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(merged_translation)
    print(f'Translation saved to {result_path}')

if __name__ == '__main__':
    asyncio.run(main())
