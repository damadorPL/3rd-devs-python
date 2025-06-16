import os
from typing import Dict, List, Any, Optional, Union
from algoliasearch.search.client import SearchClient
from vector_service import VectorService

class SearchService:
    def __init__(self, application_id: str, api_key: str):
        self.client = SearchClient(application_id, api_key)

    async def search_single_index(
        self,
        index_name: str,
        query: str,
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        default_params = {
            'hitsPerPage': 20,
            'page': 0,
            'attributesToRetrieve': ['*'],
            'typoTolerance': True,
            'ignorePlurals': True,
            'removeStopWords': False,
            'queryType': 'prefixAll', # Changed from 'prefixNone' to allow partial matches
            'attributesToHighlight': ['*'],
            'highlightPreTag': '<em>',
            'highlightPostTag': '</em>',
            'analytics': True,
            'clickAnalytics': True,
            'enablePersonalization': False,
            'distinct': 1,
            'facets': ['*'],
            'minWordSizefor1Typo': 3,  # Lowered from 4 to allow typos on shorter words
            'minWordSizefor2Typos': 7,  # Lowered from 8 for consistency
            'advancedSyntax': True,
            'removeWordsIfNoResults': 'none' # Changed from 'lastWords' to prevent word removal
        }

        merged_params = {
            **default_params,
            **(options.get('queryParameters', {}) if options else {}),
            'query': query,
            'optionalWords': query,
            'indexName': index_name,
            'getRankingInfo': True
        }

        search_params = {"requests":[merged_params]}
        headers = options.get("headers", {}) if options else {}

        search_results = await self.client.search(
            search_method_params = search_params,
            request_options=headers if headers else None
        )

        # Transform Algolia results to match document structure
        return [
            {k: v for k, v in hit.model_extra.items() if k not in ['object_id', 'highlight_result', 'ranking_info']}
            for hit in search_results.results[0].actual_instance.hits
        ]

    async def save_object(self, index_name: str, obj: Dict[str, Any]) -> Any:
        obj_with_id = {**obj, 'objectID': obj['uuid']}
        return await self.client.save_object(index_name, obj_with_id)

    async def save_objects(self, index_name: str, objects: List[Dict[str, Any]]) -> Any:
        objects_with_id = [{**obj, 'objectID': obj['uuid']} for obj in objects]
        return await self.client.save_objects(index_name, {'objects': objects_with_id})

    async def get_object(
        self,
        index_name: str,
        object_id: str,
        attributes_to_retrieve: Optional[List[str]] = None
    ) -> Any:
        return await self.client.get_object(
            index_name,
            object_id,
            attributes_to_retrieve=attributes_to_retrieve
        )

    async def partial_update_object(
        self,
        index_name: str,
        object_id: str,
        attributes: Dict[str, Any]
    ) -> Any:
        return await self.client.partial_update_object(
            index_name,
            object_id,
            attributes_to_update=attributes
        )

    async def delete_object(self, index_name: str, object_id: str) -> Any:
        return await self.client.delete_object(index_name, object_id)

    async def delete_by(self, index_name: str, filters: str) -> Any:
        return await self.client.delete_by(index_name, delete_by_params={'filters': filters})

    async def clear_objects(self, index_name: str) -> Any:
        return await self.client.clear_objects(index_name)

    async def get_objects(
        self,
        index_name: str,
        object_ids: List[str],
        attributes_to_retrieve: Optional[List[str]] = None
    ) -> Any:
        requests = [
            {'indexName': index_name, 'objectID': object_id, 'attributesToRetrieve': attributes_to_retrieve}
            for object_id in object_ids
        ]
        return await self.client.get_objects({'requests': requests})

    async def list_indices(self) -> Any:
        return await self.client.list_indices()
