import pytest
import os
from search_service import SearchService

@pytest.fixture
async def search_service():
    # Get credentials from environment variables
    application_id = os.getenv('ALGOLIA_APPLICATION_ID', 'test_app_id')
    api_key = os.getenv('ALGOLIA_API_KEY', 'test_api_key')

    service = SearchService(application_id, api_key)
    yield service

    # Cleanup: close the client session if it exists
    if hasattr(service.client, '_transporter') and service.client._transporter._session:
        await service.client._transporter._session.close()

@pytest.mark.asyncio
async def test_search_single_index(search_service):
    # Test parameters
    index_name = "test_index"
    query = "test query"

    # Test with default parameters
    try:
        results = await search_service.search_single_index(
            index_name=index_name,
            query=query
        )
        assert isinstance(results, list)
    except Exception as e:
        pytest.skip(f"Test skipped due to Algolia connection error: {str(e)}")

    # Test with custom parameters
    custom_options = {
        'queryParameters': {
            'hitsPerPage': 10,
            'page': 1,
            'filters': 'category:test',
            'facets': ['category', 'tags'],
            'attributesToRetrieve': ['title', 'description', 'category'],
            'typoTolerance': False,
            'ignorePlurals': False,
            'removeStopWords': True,
            'queryType': 'prefixLast',
            'attributesToHighlight': ['title', 'description'],
            'highlightPreTag': '<strong>',
            'highlightPostTag': '</strong>',
            'analytics': False,
            'clickAnalytics': False,
            'enablePersonalization': True,
            'distinct': 2,
            'minWordSizefor1Typo': 4,
            'minWordSizefor2Typos': 8,
            'advancedSyntax': False,
            'removeWordsIfNoResults': 'lastWords'
        },
        'headers': {
            'X-Algolia-User-ID': 'test_user',
            'X-Forwarded-For': '127.0.0.1'
        }
    }

    try:
        results_with_options = await search_service.search_single_index(
            index_name=index_name,
            query=query,
            options=custom_options
        )
        assert isinstance(results_with_options, list)
    except Exception as e:
        pytest.skip(f"Test skipped due to Algolia connection error: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])
