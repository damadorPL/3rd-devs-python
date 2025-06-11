import json
import re
import base64
import asyncio
from typing import List, Dict
import requests
from pathlib import Path

from openai_service import OpenAIService
from prompts import (
    extract_image_context_system_message,
    refine_description_system_message,
    preview_image_system_message
)

class Image:
    def __init__(self, alt: str, url: str, name: str, base64_data: str):
        self.alt = alt
        self.url = url
        self.context = ''
        self.description = ''
        self.preview = ''
        self.base64 = base64_data
        self.name = name

openai_service = OpenAIService()

async def extract_images(article: str) -> List[Image]:
    image_regex = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(image_regex, article)
    
    images = []
    for alt, url in matches:
        try:
            name = url.split('/')[-1] if '/' in url else ''
            response = requests.get(url)
            
            if not response.ok:
                raise Exception(f"Failed to fetch {url}: {response.status_text}")
            
            base64_data = base64.b64encode(response.content).decode('utf-8')
            
            images.append(Image(alt, url, name, base64_data))
            
        except Exception as error:
            print(f"Error processing image {url}: {error}")
            continue
    
    return images

async def preview_image(image: Image) -> Dict[str, str]:
    user_message = {
        'role': 'user',
        'content': [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image.base64}"}
            },
            {
                "type": "text",
                "text": f"Describe the image {image.name} concisely. Focus on the main elements and overall composition. Return the result in JSON format with only 'name' and 'preview' properties."
            }
        ]
    }
    
    response = openai_service.completion(
        [preview_image_system_message, user_message],
        'gpt-4o',
        False,
        True
    )
    
    result = json.loads(response.choices[0].message.content or '{}')
    return {'name': result.get('name', image.name), 'preview': result.get('preview', '')}

async def get_image_context(title: str, article: str, images: List[Image]) -> Dict[str, List[Dict[str, str]]]:
    user_message = {
        'role': 'user',
        'content': f"Title: {title}\n\n{article}"
    }
    
    image_refs = [{'name': img.name, 'url': img.url} for img in images]
    
    response = openai_service.completion(
        [extract_image_context_system_message(image_refs), user_message],
        'gpt-4o',
        False,
        True
    )
    
    result = json.loads(response.choices[0].message.content or '{}')
    
    # Generate previews for all images simultaneously
    preview_tasks = [preview_image(image) for image in images]
    previews = await asyncio.gather(*preview_tasks)
    
    # Merge context and preview information
    merged_results = []
    for context_image in result.get('images', []):
        preview = next((p for p in previews if p['name'] == context_image['name']), None)
        merged_results.append({
            **context_image,
            'preview': preview['preview'] if preview else ''
        })
    
    return {'images': merged_results}

async def refine_description(image: Image) -> Image:
    user_message = {
        'role': 'user',
        'content': [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image.base64}"}
            },
            {
                "type": "text",
                "text": f"Write a description of the image {image.name}. I have some {image.context} that should be useful for understanding the image in a better way. An initial preview of the image is: {image.preview}. A good description briefly describes what is on the image, and uses the context to make it more relevant to the article. The purpose of this description is for summarizing the article, so we need just an essence of the image considering the context, not a detailed description of what is on the image."
            }
        ]
    }
    
    print(user_message)
    
    response = openai_service.completion(
        [refine_description_system_message, user_message],
        'gpt-4o',
        False
    )
    
    result = response.choices[0].message.content or ''
    image.description = result
    return image

async def process_and_summarize_images(title: str, path: str):
    """
    Generates a detailed summary by orchestrating all processing steps, 
    including embedding relevant links and images within the content.
    """
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Read the article file - resolve path relative to script location
    article_path = current_dir / path
    with open(article_path, 'r', encoding='utf-8') as file:
        article = file.read()
    
    # Extract images from the article
    images = await extract_images(article)
    print(f'Number of images found: {len(images)}')
    
    contexts = await get_image_context(title, article, images)
    print(f'Number of image metadata found: {len(contexts["images"])}')
    
    # Process each image: use context and preview from get_image_context, then refine description
    processed_images = []
    for image in images:
        context_data = next(
            (ctx for ctx in contexts['images'] if ctx['name'] == image.name),
            {'context': '', 'preview': ''}
        )
        
        image.context = context_data.get('context', '')
        image.preview = context_data.get('preview', '')
        
        refined_image = await refine_description(image)
        processed_images.append(refined_image)
    
    # Prepare and save the summarized images (excluding base64 data)
    described_images = []
    for img in processed_images:
        described_images.append({
            'alt': img.alt,
            'url': img.url,
            'context': img.context,
            'description': img.description,
            'preview': img.preview,
            'name': img.name
        })
    
    # Save descriptions
    descriptions_path = current_dir / 'descriptions.json'
    with open(descriptions_path, 'w', encoding='utf-8') as file:
        json.dump(described_images, file, indent=2, ensure_ascii=False)
    
    # Prepare and save the final data (only url and description)
    captions = [{'url': img['url'], 'description': img['description']} for img in described_images]
    
    captions_path = current_dir / 'captions.json'
    with open(captions_path, 'w', encoding='utf-8') as file:
        json.dump(captions, file, indent=2, ensure_ascii=False)
    
    # Log completion messages
    print(f'Descriptions saved to {descriptions_path}')
    print(f'Captions saved to {captions_path}')

# Execute the main function
if __name__ == "__main__":
    asyncio.run(
        process_and_summarize_images(
            'Lesson #0201 — Audio i interfejs głosowy',
            'article.md'
        )
    )
