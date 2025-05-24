from typing import List, Dict, Any, TypedDict
from OpenAIService import OpenAIService
from prompts import (
    use_search_prompt,
    ask_domains_prompt,
    score_results_prompt,
    select_resources_to_load_prompt
)
import os
import json
from urllib.parse import urlparse
import aiohttp
from firecrawl import FirecrawlApp

class SearchNecessityResponse(TypedDict):
    score: float
    reason: str

class WebSearchService:
    def __init__(self, allowed_domains: List[Dict[str, str]], api_key: str):
        self.allowed_domains = allowed_domains
        self.api_key = api_key
        self.firecrawl = FirecrawlApp(api_key=api_key)
        self.openai_service = OpenAIService()

    async def is_web_search_needed(self, user_input: str) -> bool:
        try:
            response = await self.openai_service.completion(
                messages=[
                    {"role": "system", "content": use_search_prompt},
                    {"role": "user", "content": user_input}
                ],
                model="gpt-4o",
                json_mode=False
            )
            # Clean the response and convert to integer
            cleaned_response = response.strip().lower()
            if cleaned_response in ['1', 'true', 'yes']:
                return True
            elif cleaned_response in ['0', 'false', 'no']:
                return False
            else:
                print(f"Unexpected response from is_web_search_needed: {response}")
                return False
        except Exception as e:
            print(f"Error in is_web_search_needed: {str(e)}")
            return False

    async def generate_queries(self, user_input: str) -> Dict[str, Any]:
        try:
            response = await self.openai_service.completion(
                messages=[
                    {"role": "system", "content": ask_domains_prompt(self.allowed_domains)},
                    {"role": "user", "content": user_input}
                ],
                model="gpt-4o",
                json_mode=True
            )
            result = json.loads(response)
            
            # Filter queries to only include allowed domains
            filtered_queries = [
                query for query in result.get("queries", [])
                if any(
                    allowed["url"] in query.get("url", "") or query.get("url", "") in allowed["url"]
                    for allowed in self.allowed_domains
                )
            ]
            
            return {
                "queries": filtered_queries,
                "thoughts": result.get("_thoughts", "")
            }
        except Exception as e:
            print(f"Error in generate_queries: {str(e)}")
            return {"queries": [], "thoughts": ""}

    async def search_web(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        try:
            results = []
            async with aiohttp.ClientSession() as session:
                for query in queries:
                    search_query = query.get("q", "")
                    domain = query.get("url", "")
                    
                    if not search_query or not domain:
                        continue

                    try:
                        # Add site: prefix to the query using domain
                        domain = urlparse(domain if domain.startswith('http') else f'https://{domain}').netloc
                        site_query = f"site:{domain} {search_query}"
                        
                        async with session.post(
                            'https://api.firecrawl.dev/v0/search',
                            headers={
                                'Content-Type': 'application/json',
                                'Authorization': f'Bearer {self.api_key}'
                            },
                            json={
                                'query': site_query,
                                'searchOptions': {
                                    'limit': 6
                                },
                                'pageOptions': {
                                    'fetchPageContent': False
                                }
                            }
                        ) as response:
                            if not response.ok:
                                raise Exception(f"HTTP error! status: {response.status}")
                            
                            result = await response.json()
                            
                            if result.get('success') and result.get('data') and isinstance(result['data'], list):
                                results.append({
                                    'query': search_query,
                                    'results': [
                                        {
                                            'url': item.get('url', ''),
                                            'title': item.get('title', ''),
                                            'description': item.get('description', '')
                                        }
                                        for item in result['data']
                                    ]
                                })
                            else:
                                print(f"No results found for query: \"{site_query}\"")
                                results.append({'query': search_query, 'results': []})
                    except Exception as e:
                        print(f"Error searching for \"{search_query}\": {str(e)}")
                        results.append({'query': search_query, 'results': []})
                        continue

            return results
        except Exception as e:
            print(f"Error in search_web: {str(e)}")
            return []

    async def score_results(self, search_results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        try:
            scored_results = []
            for result in search_results:
                for item in result['results']:
                    try:
                        user_message = f"""<context>
                        Resource: {item['url']}
                        Snippet: {item['description']}
                        </context>

                        The following is the original user query that we are scoring the resource against. It's super relevant.
                        <original_user_query_to_consider>
                        {original_query}
                        </original_user_query_to_consider>

                        The following is the generated query that may be helpful in scoring the resource.
                        <query>
                        {result['query']}
                        </query>"""

                        response = await self.openai_service.completion(
                            messages=[
                                {"role": "system", "content": score_results_prompt},
                                {"role": "user", "content": user_message}
                            ],
                            model="gpt-4o-mini",
                            json_mode=True
                        )
                        score_data = json.loads(response)
                        item['relevance_score'] = score_data.get('score', 0)
                        item['relevance_reason'] = score_data.get('reason', '')
                        scored_results.append(item)
                    except Exception as e:
                        print(f"Error scoring result: {str(e)}")
                        continue

            # Sort by score and take top 3
            scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            return scored_results[:3]
        except Exception as e:
            print(f"Error in score_results: {str(e)}")
            return []

    async def select_resources_to_load(self, user_input: str, filtered_results: List[Dict[str, Any]]) -> List[str]:
        try:
            response = await self.openai_service.completion(
                messages=[
                    {"role": "system", "content": select_resources_to_load_prompt},
                    {"role": "user", "content": f"Original query: \"{user_input}\"\nFiltered resources:\n{json.dumps([{'url': r['url'], 'snippet': r['description']} for r in filtered_results], indent=2)}"}
                ],
                model="gpt-4o",
                json_mode=True
            )
            result = json.loads(response)
            selected_urls = result.get('urls', [])
            
            # Filter out URLs that aren't in the filtered results
            valid_urls = [
                url for url in selected_urls
                if any(r['url'] == url for r in filtered_results)
            ]
            
            return valid_urls
        except Exception as e:
            print(f"Error in select_resources_to_load: {str(e)}")
            return []

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        try:
            # Filter out URLs that are not scrappable based on allowed_domains
            scrappable_urls = [
                url for url in urls
                if any(
                    allowed['url'] == urlparse(url).netloc.replace('www.', '')
                    and allowed['scrappable']
                    for allowed in self.allowed_domains
                )
            ]

            scraped_results = []
            for url in scrappable_urls:
                try:
                    scrape_result = await self.firecrawl.scrape(url=url, formats=['markdown'])
                    
                    if scrape_result and scrape_result.get('markdown'):
                        scraped_results.append({
                            'url': url,
                            'content': scrape_result['markdown']
                        })
                    else:
                        print(f"No markdown content found for URL: {url}")
                except Exception as e:
                    print(f"Error scraping URL {url}: {str(e)}")
                    continue

            return scraped_results
        except Exception as e:
            print(f"Error in scrape_urls: {str(e)}")
            return [] 