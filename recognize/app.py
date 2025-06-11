import asyncio
import os
import base64
from pathlib import Path
from typing import Dict
from openai_service import OpenAIService

async def process_avatar(file: str, appearance_description: str, openai_service: OpenAIService) -> Dict[str, str]:
    """Process a single avatar file and get AI response"""
    avatar_folder = Path(__file__).parent / 'avatars'
    file_path = avatar_folder / file
    
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    base64_image = base64.b64encode(file_data).decode('utf-8')
    
    messages = [
        {
            "role": "system",
            "content": f"As Alice, you need to use a description of how you look and write back with \"it's me\" or \"it's not me\" when the user sends you a photo of yourself.\n{appearance_description}"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "Is that you?"
                }
            ]
        }
    ]
    
    chat_completion = await openai_service.completion(messages, "gpt-4o", False, False, 1024)
    
    return {
        "file": file,
        "response": chat_completion.choices[0].message.content or ''
    }

async def process_avatars():
    """Main function to process all avatar images"""
    avatar_folder = Path(__file__).parent / 'avatars'
    files = [f for f in os.listdir(avatar_folder) if f.endswith('.png')]
    
    appearance_description = ("""
        I have long, flowing dark hair with striking purple highlights that catch the light beautifully.
        I have intense, captivating eyes framed by bold, smoky eye makeup that really makes them pop.
        I have high, defined cheekbones and full, plump lips that give my face a strong, confident structure.
        I have smooth, flawless skin with a warm, olive complexion that glows in the golden light.
        I have a strong jawline that adds to my bold appearance.
        I have on a dark, casual hoodie that contrasts nicely with my dramatic features, balancing out my edgy yet glamorous look.
    """)
    
    openai_service = OpenAIService()
    
    # Process all files concurrently
    tasks = [process_avatar(file, appearance_description, openai_service) for file in files]
    results = await asyncio.gather(*tasks)
    
    # Display results in table format (similar to console.table)
    print(f"{'Index':<6} {'File':<30} {'Response'}")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"{i:<6} {result['file']:<30} {result['response']}")

if __name__ == "__main__":
    asyncio.run(process_avatars())
