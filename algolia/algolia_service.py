from algoliasearch.search.client import SearchClient
from typing import Dict, List, Optional, Any

class AlgoliaService:
    def __init__(self, application_id: str, api_key: str):
        self.client = SearchClient(application_id, api_key)
    
    async def search_single_index(self, index_name: str, query: str, options: Optional[Dict] = None):
        default_params = {
            "hitsPerPage": 20,
            "page": 0,
            "attributesToRetrieve": ["*"],
            "typoTolerance": True,
            "ignorePlurals": True,
            "removeStopWords": True,
            "queryType": "prefixNone",
            "attributesToHighlight": ["*"],
            "highlightPreTag": "<em>",
            "highlightPostTag": "</em>",
            "analytics": True,
            "clickAnalytics": True,
            "enablePersonalization": False,
            "distinct": 1,
            "facets": ["*"],
            "minWordSizefor1Typo": 1,
            "minWordSizefor2Typos": 3,
            "advancedSyntax": True,
            "removeWordsIfNoResults": "lastWords",
            "getRankingInfo": True
        }
        
        merged_params = {
            **default_params,
            "query": query
        }
        
        if options and "queryParameters" in options:
            merged_params.update(options["queryParameters"])
        
        # Use correct Python search API structure
        search_params = {
            "requests": [{
                "indexName": index_name,
                **merged_params
            }]
        }
        
        headers = options.get("headers", {}) if options else {}
        
        return await self.client.search(
            search_method_params=search_params,
            request_options=headers if headers else None
        )
    
    async def save_object(self, index_name: str, object: Dict[str, Any]):
        return await self.client.save_object(index_name=index_name, body=object)
    
    async def get_object(self, index_name: str, object_id: str, attributes_to_retrieve: Optional[List[str]] = None):
        params = {"index_name": index_name, "object_id": object_id}
        if attributes_to_retrieve:
            params["attributes_to_retrieve"] = attributes_to_retrieve
        return await self.client.get_object(**params)
    
    async def add_or_update_object(self, index_name: str, object_id: str, object: Dict[str, Any]):
        return await self.client.save_object(
            index_name=index_name, 
            body={**object, "objectID": object_id}
        )
    
    async def delete_object(self, index_name: str, object_id: str):
        return await self.client.delete_object(index_name=index_name, object_id=object_id)
    
    async def delete_by(self, index_name: str, filters: str):
        return await self.client.delete_by(
            index_name=index_name, 
            delete_by_params={"filters": filters}
        )
    
    async def clear_objects(self, index_name: str):
        return await self.client.clear_objects(index_name=index_name)
    
    async def partial_update_object(self, index_name: str, object_id: str, attributes: Dict[str, Any]):
        return await self.client.partial_update_object(
            index_name=index_name,
            object_id=object_id,
            attributes_to_update=attributes
        )
    
    async def get_objects(self, requests: List[Dict[str, Any]]):
        return await self.client.get_objects(requests=requests)
    
    async def list_indices(self):
        return await self.client.list_indices()
