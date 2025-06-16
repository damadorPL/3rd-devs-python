import os
import json
import aiohttp
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Union, TypedDict
from urllib.parse import urlparse, urljoin
from openai_service import OpenAIService
from prompts.pick_resources import get_prompt as select_resources_to_load_prompt, SearchResult
from firecrawl import FirecrawlApp

class ChatCompletionMessageParam(TypedDict):
    role: str
    content: str

class WebSearchService:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.allowed_domains = [
            {'name': 'Wikipedia', 'url': 'wikipedia.org', 'scrappable': True},
            {'name': 'easycart', 'url': 'easycart.pl', 'scrappable': True},
            {'name': 'FS.blog', 'url': 'fs.blog', 'scrappable': True},
            {'name': 'arXiv', 'url': 'arxiv.org', 'scrappable': True},
            {'name': 'Instagram', 'url': 'instagram.com', 'scrappable': False},
            {'name': 'OpenAI', 'url': 'openai.com', 'scrappable': True},
            {'name': 'Brain overment', 'url': 'brain.overment.com', 'scrappable': True},
            {'name': 'Reuters', 'url': 'reuters.com', 'scrappable': True},
            {'name': 'MIT Technology Review', 'url': 'technologyreview.com', 'scrappable': True},
            {'name': 'Youtube', 'url': 'youtube.com', 'scrappable': False},
            {'name': 'Mrugalski / UWteam', 'url': 'mrugalski.pl', 'scrappable': True},
            {'name': 'Hacker News', 'url': 'news.ycombinator.com', 'scrappable': True},
        ]
        self.api_key = os.getenv('FIRECRAWL_API_KEY', '')
        self.firecrawl_app = FirecrawlApp(api_key=self.api_key)

    async def search_web(self, queries: List[Dict[str, str]], conversation_uuid: str) -> List[Dict[str, Any]]:
        search_results = []
        for query_info in queries:
            q, url = query_info['q'], query_info['url']
            try:
                domain = urlparse(url if url.startswith('http') else f'https://{url}').netloc
                site_query = f'site:{domain} {q}'
                print('siteQuery', site_query)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'https://api.firecrawl.dev/v0/search',
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {self.api_key}'
                        },
                        json={
                            'query': site_query,
                            'searchOptions': {'limit': 6},
                            'pageOptions': {'fetchPageContent': False}
                        }
                    ) as response:
                        if response.status != 200:
                            raise Exception(f'HTTP error! status: {response.status}')
                        result = await response.json()
                        if result.get('success') and result.get('data'):
                            search_results.append({
                                'query': q,
                                'domain': domain,
                                'results': [{'url': item['url'], 'title': item['title'], 'description': item['description']} for item in result['data']]
                            })
                        else:
                            print(f'No results found for query: "{site_query}"')
                            search_results.append({'query': q, 'domain': domain, 'results': []})
            except Exception as error:
                print(f'Error searching for "{q}":', error)
                search_results.append({'query': q, 'domain': url, 'results': []})
        return search_results

    async def select_resources_to_load(
        self,
        messages: List[ChatCompletionMessageParam],
        filtered_results: List[SearchResult]
    ) -> List[str]:
        try:
            # Get the prompt with the filtered results
            prompt = select_resources_to_load_prompt(filtered_results)

            # Create system message with the prompt
            system_prompt = {'role': 'system', 'content': prompt}

            # Get response from the model
            response = await self.openai_service.completion([system_prompt] + messages, 'gpt-4o', False, True)

            if response.choices[0].message.content:
                result = json.loads(response.choices[0].message.content)
                selected_urls = result.get('urls', [])
                valid_urls = [url for url in selected_urls if any(r['results'] and any(item['url'] == url for item in r['results']) for r in filtered_results)]
                empty_domains = [r['domain'] for r in filtered_results if not r['results']]
                combined_urls = valid_urls + empty_domains
                return combined_urls
            raise Exception('Unexpected response format')
        except Exception as error:
            print('Error selecting resources to load:', error)
            return []

    async def scrape_urls(self, urls: List[str], conversation_uuid: str) -> List[Dict[str, str]]:
        print('Input (scrapeUrls):', urls)
        scrappable_urls = [url for url in urls if self.is_scrappable(url)]
        scrape_results = []
        for url in scrappable_urls:
            try:
                url = url.rstrip('/')
                scrape_result = await self.firecrawl_app.scrape_url(url, formats=['markdown'])
                if scrape_result and hasattr(scrape_result, 'markdown'):
                    scrape_results.append({
                        'url': url,
                        'content': scrape_result.markdown.strip()
                    })
                else:
                    print(f'No markdown content found for URL: {url}')
                    scrape_results.append({'url': url, 'content': ''})
            except Exception as error:
                print(f'Error scraping URL {url}:', error)
                scrape_results.append({'url': url, 'content': ''})
        return [result for result in scrape_results if result['content']]

    def is_scrappable(self, url: str) -> bool:
        domain = urlparse(url).hostname.replace('www.', '')
        allowed_domain = next((d for d in self.allowed_domains if d['url'] == domain), None)
        return allowed_domain and allowed_domain['scrappable']
