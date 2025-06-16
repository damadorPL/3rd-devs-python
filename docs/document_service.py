import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai_service import OpenAIService
from text_service import TextService, IDoc
from database_service import DatabaseService
from prompts.extract import get_prompt as extract_prompt
from prompts.translate import get_prompt as translate_prompt
from prompts.queries import get_prompt as queries_prompt
from prompts.answer import get_prompt as answer_prompt
from prompts.compress import get_prompt as compress_prompt
from prompts.synthesize import get_prompt as synthesize_prompt
from utils import get_result

class DocumentService:
    def __init__(
        self,
        openai_service: OpenAIService,
        database_service: DatabaseService,
        text_service: TextService
    ):
        self.openai_service = openai_service
        self.database_service = database_service
        self.text_service = text_service

    async def _ensure_directory_exists(self, file_path: str) -> None:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

    async def answer(self, query: str, documents: List[IDoc]) -> str:
        if not documents:
            return "No documents found"

        try:
            generated_queries = await self.openai_service.completion(
                messages=[
                    {"role": "system", "content": queries_prompt()},
                    {"role": "user", "content": query}
                ],
                model="gpt-4o",
                stream=False,
                json_mode=True
            )

            queries = json.loads(generated_queries.choices[0].message.content or "[]").get('queries', [])

            if not queries:
                return "No queries found"

            # Gather unique source_uuids
            source_uuids = {doc.metadata.get('source_uuid') for doc in documents}

            # Insert documents that DON'T exist in the database
            insert_tasks = []
            for doc in documents:
                if not doc.metadata.get('uuid'):
                    continue
                existing_doc = await self.database_service.get_document_by_uuid(doc.metadata['uuid'])
                if not existing_doc:
                    insert_tasks.append(self.database_service.insert_document(doc, True))
            await asyncio.gather(*insert_tasks)

            # Prepare filters for hybrid search
            vector_filter = {
                'should': [{'key': 'source_uuid', 'match': {'value': uuid}} for uuid in source_uuids]
            }
            fulltext_filter = {
                'queryParameters': {
                    'filters': ' OR '.join(f'source_uuid:{uuid}' for uuid in source_uuids)
                }
            }

            # Gather hybrid search results for all queries
            hybrid_results = []
            for query_item in queries:
                results = await self.database_service.hybrid_search(
                    {'query': query_item['natural'], 'filter': vector_filter},
                    {'query': query_item['search'], 'filter': fulltext_filter}
                )
                hybrid_results.extend([
                    {**doc, 'metadata': {**doc['metadata'], 'query': query_item['natural']}}
                    for doc in results
                ])

            results = [self.text_service.restore_placeholders(doc) for doc in hybrid_results]

            context = '\n'.join(
                f'<doc uuid="{doc.metadata["uuid"]}" source-uuid="{doc.metadata["source_uuid"]}" '
                f'name="{doc.metadata.get("name", "")}" query="{doc.metadata["query"]}">{doc.text}</doc>'
                for doc in results
            )

            generated_answer = await self.openai_service.completion(
                messages=[
                    {"role": "system", "content": answer_prompt({'context': context})},
                    {"role": "user", "content": query}
                ],
                model="gpt-4o",
                stream=False
            )

            answer = get_result(generated_answer.choices[0].message.content or "", "final_answer")
            return answer or "No answer found"
        except Exception as error:
            print('Error in answer method:', error)
            return "Error processing answer"

    async def synthesize(self, query: str, documents: List[IDoc]) -> str:
        if not documents:
            return "No documents found"

        try:
            processed_docs = [self.text_service.restore_placeholders(doc) for doc in documents]
            previous_answer = ""

            for doc in processed_docs:
                messages = [
                    {
                        "role": "system",
                        "content": synthesize_prompt({
                            'previousAnswer': previous_answer,
                            'originalQuery': query
                        })
                    },
                    {
                        "role": "user",
                        "content": f"Refine your answer using the following information:\n\n{doc.text}"
                    }
                ]

                completion = await self.openai_service.completion(
                    messages=messages,
                    model="gpt-4o",
                    stream=False
                )

                previous_answer = get_result(completion.choices[0].message.content or "", "final_answer") or ""

            return previous_answer or "No synthesis generated"
        except Exception as error:
            print('Error in synthesize method:', error)
            return "Error processing synthesis"

    async def summarize(self, documents: List[IDoc], general_context: Optional[str] = None) -> str:
        try:
            # Process all documents in parallel while preserving order
            compression_tasks = []
            for doc in documents:
                try:
                    completion = await self.openai_service.completion(
                        messages=[
                            {"role": "system", "content": compress_prompt(general_context)},
                            {"role": "user", "content": doc.text}
                        ],
                        model="gpt-4o",
                        max_tokens=10000,
                        stream=False
                    )
                    result = completion.choices[0].message.content
                    if not result:
                        print('Empty completion result for document')
                        compression_tasks.append('')
                    else:
                        compression_tasks.append(result)
                except Exception as error:
                    print('Error compressing document:', error)
                    compression_tasks.append('')

            # Process results while maintaining document order
            processed_docs = []
            for doc, compressed_text in zip(documents, compression_tasks):
                processed_doc = self.text_service.restore_placeholders(
                    IDoc(
                        text=compressed_text or '',
                        metadata=doc.metadata
                    )
                )
                processed_docs.append(processed_doc)

            # Merge all compressed content
            merged_content = '\n\n'.join(doc.text for doc in processed_docs if doc.text)

            if not merged_content:
                print('No content generated after compression')
                return 'No content generated'

            try:
                # Save to file
                compression_path = os.path.join(os.path.dirname(__file__), 'results', 'compression.md')
                await self._ensure_directory_exists(compression_path)
                with open(compression_path, 'w', encoding='utf-8') as f:
                    f.write(merged_content)
                print('Content saved to:', compression_path)
            except Exception as error:
                print('Error saving file:', error)

            return merged_content
        except Exception as error:
            print('Error in summarize method:', error)
            return "Error processing summary"

    async def extract(
        self,
        documents: List[IDoc],
        type: str,
        description: str,
        context: Optional[str] = None
    ) -> List[IDoc]:
        try:
            batch_size = 5
            extracted_docs = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_tasks = []
                for doc in batch:
                    messages = [
                        {
                            "role": "system",
                            "content": extract_prompt({
                                'type': type,
                                'description': description,
                                'context': context or doc.metadata.get('name', '')
                            })
                        },
                        {"role": "user", "content": doc.text}
                    ]

                    completion = await self.openai_service.completion(
                        messages=messages,
                        model="gpt-4o",
                        stream=False
                    )

                    extracted_content = get_result(completion.choices[0].message.content or "", "final_answer")
                    extracted_doc = IDoc(
                        text=extracted_content or "No results",
                        metadata={
                            **doc.metadata,
                            'extracted_type': type
                        }
                    )
                    batch_tasks.append(extracted_doc)

                extracted_docs.extend(batch_tasks)

            return [self.text_service.restore_placeholders(doc) for doc in extracted_docs]
        except Exception as error:
            print('Error in extract method:', error)
            return []

    async def translate(
        self,
        documents: List[IDoc],
        source_language: str,
        target_language: str
    ) -> List[IDoc]:
        try:
            batch_size = 5
            translated_docs = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_tasks = []
                for doc in batch:
                    messages = [
                        {"role": "system", "content": translate_prompt()},
                        {
                            "role": "user",
                            "content": f"Translate the following text from {source_language} to {target_language}:\n\n{doc.text}"
                        }
                    ]

                    completion = await self.openai_service.completion(
                        messages=messages,
                        model="gpt-4o",
                        stream=False
                    )

                    translated_content = completion.choices[0].message.content or ""
                    translated_doc = IDoc(
                        text=translated_content,
                        metadata={
                            **doc.metadata,
                            'translated_from': source_language,
                            'translated_to': target_language
                        }
                    )
                    batch_tasks.append(translated_doc)

                translated_docs.extend(batch_tasks)

            return [self.text_service.restore_placeholders(doc) for doc in translated_docs]
        except Exception as error:
            print('Error in translate method:', error)
            return []
