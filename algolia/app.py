import asyncio
import os
from uuid import uuid4
from tabulate import tabulate
from algolia_service import AlgoliaService

algolia_service = AlgoliaService(
    str(os.getenv("ALGOLIA_APP_ID")), 
    str(os.getenv("ALGOLIA_API_KEY"))
)

data = [
    {
        "author": "Adam",
        "text": "I believe in writing clean, maintainable code. Refactoring should be a regular part of our development process."
    },
    {
        "author": "Kuba", 
        "text": "Test-driven development has significantly improved the quality of our codebase. Let's make it a standard practice."
    },
    {
        "author": "Mateusz",
        "text": "Optimizing our CI/CD pipeline could greatly enhance our deployment efficiency. We should prioritize this in our next sprint."
    }
]

index_name = "dev_comments"

async def main():
    try:
        indices = await algolia_service.list_indices()
        index_exists = any(index.name == index_name for index in indices.items)
        
        if not index_exists:
            for item in data:
                object_id = str(uuid4())
                await algolia_service.add_or_update_object(index_name, object_id, {**item, "objectID": object_id})
            
            print("Data added to index")
        else:
            print("Index already exists. Skipping data addition.")
        
        query = "code"
        search_result = await algolia_service.search_single_index(index_name, query, {
            "queryParameters": {
                "filters": "author:Adam"
            }
        })
        
        hits = search_result.results[0].actual_instance.hits
        
        table_data = []
        for hit in hits:
            table_data.append({
                "Author": hit.author,
                "Text": hit.text[:45] + ("..." if len(hit.text) > 45 else ""),
                "ObjectID": hit.object_id,
                "MatchLevel": hit.highlight_result['text'].actual_instance.match_level,
                "MatchedWords": ", ".join(hit.highlight_result['text'].actual_instance.matched_words),
                "UserScore": hit.ranking_info.user_score
            })
        
        if table_data:
            headers = ["Author", "Text", "ObjectID", "MatchLevel", "MatchedWords", "UserScore"]
            formatted_data = [[row[header] for header in headers] for row in table_data]
            print(tabulate(formatted_data, headers=headers, tablefmt="grid"))
        else:
            print("No results found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
