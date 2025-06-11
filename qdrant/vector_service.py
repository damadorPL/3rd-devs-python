import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai_service import OpenAIService

class VectorService:
    def __init__(self, openai_service: OpenAIService):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.openai_service = openai_service
    
    def ensure_collection(self, name: str):
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if name not in collection_names:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
    
    def add_points(self, collection_name: str, points: List[Dict[str, Any]]):
        points_to_upsert = []
        
        for point in points:
            embedding = self.openai_service.create_embedding(point["text"])
            points_to_upsert.append(
                PointStruct(
                    id=point["id"],
                    vector=embedding,
                    payload={"text": point["text"], "role": point["role"]}
                )
            )
        
        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points_to_upsert
        )
    
    def perform_search(self, collection_name: str, query: str, limit: int = 5):
        query_embedding = self.openai_service.create_embedding(query)
        
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
