import os
import uuid
from typing import List, Dict, Any, Optional, TypedDict, Union
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from openai_service import OpenAIService

class Point(TypedDict, total=False):
    id: str
    text: str
    metadata: Dict[str, Any]

class VectorService:
    def __init__(self, openai_service: OpenAIService):
        self.client = AsyncQdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        self.openai_service = openai_service

    async def ensure_collection(self, name: str) -> None:
        collections = await self.client.get_collections()
        if not any(c.name == name for c in collections.collections):
            await self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=3072,
                    distance=models.Distance.COSINE
                )
            )

    async def add_points(
        self,
        collection_name: str,
        points: List[Point]
    ) -> None:
        await self.ensure_collection(collection_name)

        # Create embeddings in parallel for better performance
        points_to_upsert = []
        for point in points:
            embedding = await self.openai_service.create_embedding(point['text'])
            points_to_upsert.append(models.PointStruct(
                id=point.get('id', str(uuid.uuid4())),
                vector=embedding,
                payload={
                    'text': point['text'],
                    **(point.get('metadata', {}))
                }
            ))

        await self.client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True
        )

    async def update_point(
        self,
        collection_name: str,
        point: Point
    ) -> None:
        await self.add_points(collection_name, [point])

    async def delete_point(self, collection_name: str, point_id: str) -> None:
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[point_id]
            ),
            wait=True
        )

    async def perform_search(
        self,
        collection_name: str,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        query_embedding = await self.openai_service.create_embedding(query)
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
            # query_filter=filter # FIXME: free version of qdrant does not support filters
        )

        # Transform Qdrant results to match document structure
        return [
            {
                'text': result.payload.get('text'),
                'metadata': result.payload
            }
            for result in results
        ]

    async def get_all_points(self, collection_name: str) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        offset = None
        has_more = True

        while has_more:
            response = await self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True
            )

            points.extend(response.points)

            if response.next_page_offset is not None:
                offset = response.next_page_offset
            else:
                has_more = False

        return points
