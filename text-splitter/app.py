from text_splitter import TextSplitter
import os
import asyncio
import json
from typing import Dict, Any

async def process_file(file_path: str) -> Dict[str, Any]:
    """Process a markdown file, split it into chunks, and generate statistics"""
    splitter = TextSplitter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    docs = await splitter.split(text, 1000)
    
    # Save to JSON
    json_file_path = os.path.splitext(file_path)[0] + '.json'
    with open(json_file_path, 'w', encoding='utf-8') as f:
        # Convert dataclasses to dict for JSON serialization
        docs_dict = [
            {
                'text': doc.text,
                'metadata': {
                    'tokens': doc.metadata.tokens,
                    'headers': doc.metadata.headers,
                    'urls': doc.metadata.urls,
                    'images': doc.metadata.images
                }
            }
            for doc in docs
        ]
        json.dump(docs_dict, f, indent=2)
    
    # Calculate statistics
    chunk_sizes = [doc.metadata.tokens for doc in docs]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
    min_chunk_size = min(chunk_sizes)
    max_chunk_size = max(chunk_sizes)
    median_chunk_size = sorted(chunk_sizes)[len(chunk_sizes) // 2]
    
    return {
        'file': os.path.basename(file_path),
        'avgChunkSize': f"{avg_chunk_size:.2f}",
        'medianChunkSize': median_chunk_size,
        'minChunkSize': min_chunk_size,
        'maxChunkSize': max_chunk_size,
        'totalChunks': len(chunk_sizes)
    }


async def main():
    """Main function to process all markdown files in directory"""
    directory_path = os.path.join(os.getcwd(), 'text-splitter')
    files = os.listdir(directory_path)
    reports = []
    
    for file in files:
        if file.endswith('.md'):
            report = await process_file(os.path.join(directory_path, file))
            reports.append(report)
    
    # Print table-like output
    if reports:
        print("File Processing Reports:")
        for report in reports:
            print(f"File: {report['file']}")
            print(f"  Avg Chunk Size: {report['avgChunkSize']}")
            print(f"  Median Chunk Size: {report['medianChunkSize']}")
            print(f"  Min Chunk Size: {report['minChunkSize']}")
            print(f"  Max Chunk Size: {report['maxChunkSize']}")
            print(f"  Total Chunks: {report['totalChunks']}")
            print()

if __name__ == "__main__":
    asyncio.run(main())