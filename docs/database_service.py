import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from search_service import SearchService
from vector_service import VectorService
from text_service import IDoc

class DatabaseService:
    def __init__(
        self,
        db_path: str = 'hybrid/database.db',
        search_service: SearchService = None,
        vector_service: VectorService = None
    ):
        self.absolute_path = Path(db_path).resolve()
        print(f"Using database at: {self.absolute_path}")

        self.db_exists = self.absolute_path.exists()
        self.conn = sqlite3.connect(str(self.absolute_path))
        self.conn.row_factory = sqlite3.Row

        self.search_service = search_service
        self.vector_service = vector_service

        if not self.db_exists:
            print('Database does not exist. Initializing...')
            self.initialize_database()
        else:
            print('Database already exists. Checking for updates...')
            self.initialize_database()

    def initialize_database(self):
        cursor = self.conn.cursor()

        # Create main documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                source_uuid TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create full-text search virtual table
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_search USING fts5(
                text, metadata,
                tokenize='porter unicode61'
            )
        ''')

        # Create triggers to keep the search index updated
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_search(rowid, text, metadata)
                VALUES (new.id, new.text, new.metadata);
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                DELETE FROM documents_search WHERE rowid = old.id;
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                UPDATE documents_search
                SET text = new.text,
                    metadata = new.metadata
                WHERE rowid = old.id;
            END
        ''')

        self.conn.commit()

    async def insert_document(self, document: IDoc, for_search: bool = False) -> Any:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO documents (uuid, source_uuid, text, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (
            document.metadata.get('uuid', ''),
            document.metadata.get('source_uuid', ''),
            document.text,
            json.dumps(document.metadata)
        ))
        self.conn.commit()

        if for_search and self.search_service and self.vector_service:
            # Sync to Algolia
            await self.search_service.save_object('documents', {
                'objectID': document.metadata.get('uuid'),
                'text': document.text,
                **document.metadata
            })

            # Sync to Qdrant
            await self.vector_service.add_points('documents', [{
                'id': document.metadata.get('uuid'),
                'text': document.text,
                'metadata': {'text': document.text, **document.metadata}
            }])

        return cursor.lastrowid

    async def update_document(self, uuid: str, document: Dict[str, Any]) -> Any:
        cursor = self.conn.cursor()

        # Only update fields that are provided
        update_fields = []
        params = []

        if document.get('text') is not None:
            update_fields.append('text = ?')
            params.append(document['text'])

        if document.get('metadata') is not None:
            update_fields.append('metadata = ?')
            params.append(json.dumps(document['metadata']))

        if document.get('uuid') is not None:
            update_fields.append('uuid = ?')
            params.append(document['uuid'])

        update_fields.append('updated_at = CURRENT_TIMESTAMP')
        params.append(uuid)

        if not update_fields:
            return 0

        query = f'''
            UPDATE documents
            SET {', '.join(update_fields)}
            WHERE uuid = ?
        '''

        cursor.execute(query, params)
        self.conn.commit()

        if self.search_service and self.vector_service:
            # Sync to Algolia
            if document.get('text') or document.get('metadata'):
                await self.search_service.partial_update_object('documents', uuid, {
                    'text': document.get('text'),
                    **(document.get('metadata', {}))
                })

            # Sync to Qdrant
            await self.vector_service.add_points('documents', [{
                'id': document.get('metadata', {}).get('uuid', uuid),
                'text': document.get('text', 'no text'),
                'metadata': {
                    'text': document.get('text', 'no text'),
                    **(document.get('metadata', {}))
                }
            }])

        return cursor.rowcount

    async def delete_document(self, uuid: str) -> Any:
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM documents WHERE uuid = ?', (uuid,))
        self.conn.commit()

        if self.search_service and self.vector_service:
            # Sync to Algolia
            await self.search_service.delete_object('documents', uuid)
            # Sync to Qdrant
            await self.vector_service.delete_point('documents', uuid)

        return cursor.rowcount

    async def get_document_by_uuid(self, uuid: str) -> Optional[IDoc]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE uuid = ?', (uuid,))
        result = cursor.fetchone()

        if result:
            return IDoc(
                text=result['text'],
                metadata=json.loads(result['metadata'])
            )
        return None

    async def get_documents_by_source_uuid(self, source_uuid: str) -> List[IDoc]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE source_uuid = ?', (source_uuid,))
        results = cursor.fetchall()

        return [
            IDoc(
                text=result['text'],
                metadata=json.loads(result['metadata'])
            )
            for result in results
        ]

    async def get_all_documents(self) -> List[IDoc]:
        print('Fetching all documents')
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents')
        results = cursor.fetchall()
        print(f"Found {len(results)} documents")

        return [
            IDoc(
                text=result['text'],
                metadata=json.loads(result['metadata'])
            )
            for result in results
        ]

    async def hybrid_search(
        self,
        vector_search: Dict[str, Any],
        fulltext_search: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Perform vector search
        vector_results = await self.vector_service.perform_search(
            'documents',
            vector_search['query'],
            vector_search.get('filter'),
            15
        )

        # Perform full-text search (Algolia)
        algolia_results = await self.search_service.search_single_index(
            'documents',
            fulltext_search['query'],
            fulltext_search.get('filter')
        )

        # Calculate RRF scores
        rrf = self._calculate_rrf(vector_results, algolia_results)
        avg_score = sum(item['score'] for item in rrf) / len(rrf) if rrf else 0
        filtered_rrf = [item for item in rrf if item['score'] >= avg_score]

        # Restructure the results
        return [
            {
                **{k: v for k, v in item.items() if k not in ['score', 'vectorRank', 'algoliaRank']},
                'metadata': {
                    'uuid': item.get('uuid'),
                    'source_uuid': item.get('source_uuid'),
                    **(item.get('metadata', {}))
                }
            }
            for item in filtered_rrf
        ]

    def _calculate_rrf(self, vector_results: List[Dict[str, Any]], algolia_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result_map = {}

        # Process vector results
        for index, result in enumerate(vector_results):
            uuid = result.get('uuid') or result.get('objectID')
            result_map[uuid] = {
                **result,
                'vectorRank': index + 1,
                'algoliaRank': float('inf')
            }

        # Process Algolia results
        for index, result in enumerate(algolia_results):
            uuid = result.get('uuid') or result.get('objectID')
            if uuid in result_map:
                result_map[uuid]['algoliaRank'] = index + 1
            else:
                result_map[uuid] = {
                    **result,
                    'vectorRank': float('inf'),
                    'algoliaRank': index + 1
                }

        # Calculate RRF scores
        k = 60  # RRF parameter
        results = []
        for result in result_map.values():
            vector_rank = result.get('vectorRank', float('inf'))
            algolia_rank = result.get('algoliaRank', float('inf'))
            score = (
                (1 / (k + vector_rank) if vector_rank != float('inf') else 0) +
                (1 / (k + algolia_rank) if algolia_rank != float('inf') else 0)
            )
            results.append({**result, 'score': score})

        # Sort by score in descending order
        return sorted(results, key=lambda x: x['score'], reverse=True)
