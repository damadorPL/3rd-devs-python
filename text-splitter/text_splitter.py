import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import tiktoken

@dataclass
class IDoc:
    text: str
    metadata: 'Metadata'

@dataclass
class Metadata:
    tokens: int
    headers: Dict[str, List[str]]
    urls: List[str]
    images: List[str]

Headers = Dict[str, List[str]]

class TextSplitter:
    def __init__(self, model_name: str = 'gpt-4'):
        self.tokenizer = None
        self.MODEL_NAME = model_name
        self.SPECIAL_TOKENS = {
            '<|im_start|>': 100264,
            '<|im_end|>': 100265,
            '<|im_sep|>': 100266,
        }
    
    async def _initialize_tokenizer(self) -> None:
        """Get or create a tokenizer for the specified model"""
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.MODEL_NAME)
            except KeyError:
                raise KeyError(f"Tokenizer for model {self.MODEL_NAME} not found.")
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text"""
        if not self.tokenizer:
            raise ValueError('Tokenizer not initialized')
        
        formatted_content = self._format_for_tokenization(text)
        tokens = self.tokenizer.encode(formatted_content)
        return len(tokens)
    
    def _format_for_tokenization(self, text: str) -> str:
        """Format text with special tokens for tokenization"""
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant<|im_end|>"
    
    async def split(self, text: str, limit: int) -> List[IDoc]:
        """Split text into chunks based on token limit"""
        print(f"Starting split process with limit: {limit} tokens")
        await self._initialize_tokenizer()
        
        chunks: List[IDoc] = []
        position = 0
        total_length = len(text)
        current_headers: Headers = {}
        
        while position < total_length:
            print(f"Processing chunk starting at position: {position}")
            chunk_text, chunk_end = self._get_chunk(text, position, limit)
            tokens = self._count_tokens(chunk_text)
            print(f"Chunk tokens: {tokens}")
            
            headers_in_chunk = self._extract_headers(chunk_text)
            self._update_current_headers(current_headers, headers_in_chunk)
            
            content, urls, images = self._extract_urls_and_images(chunk_text)
            
            chunks.append(IDoc(
                text=content,
                metadata=Metadata(
                    tokens=tokens,
                    headers=current_headers.copy(),
                    urls=urls,
                    images=images
                )
            ))
            
            print(f"Chunk processed. New position: {chunk_end}")
            position = chunk_end
        
        print(f"Split process completed. Total chunks: {len(chunks)}")
        return chunks
    
    def _get_chunk(self, text: str, start: int, limit: int) -> Tuple[str, int]:
        """Extract a chunk of text starting from a position with token limit"""
        print(f"Getting chunk starting at {start} with limit {limit}")
        
        # Account for token overhead due to formatting
        overhead = self._count_tokens(self._format_for_tokenization('')) - self._count_tokens('')
        
        # Initial tentative end position
        remaining_text = text[start:]
        if remaining_text:
            estimated_tokens = self._count_tokens(remaining_text)
            end = min(start + int((len(remaining_text) * limit) / estimated_tokens), len(text))
        else:
            end = len(text)
        
        # Adjust end to avoid exceeding token limit
        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)
        
        while tokens + overhead > limit and end > start:
            print(f"Chunk exceeds limit with {tokens + overhead} tokens. Adjusting end position...")
            end = self._find_new_chunk_end(text, start, end)
            chunk_text = text[start:end]
            tokens = self._count_tokens(chunk_text)
        
        # Adjust chunk end to align with newlines
        end = self._adjust_chunk_end(text, start, end, tokens + overhead, limit)
        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)
        
        print(f"Final chunk end: {end}")
        return chunk_text, end
    
    def _adjust_chunk_end(self, text: str, start: int, end: int, current_tokens: int, limit: int) -> int:
        """Adjust chunk end position to align with newlines"""
        min_chunk_tokens = limit * 0.8  # Minimum chunk size is 80% of limit
        
        next_newline = text.find('\n', end)
        prev_newline = text.rfind('\n', start, end)
        
        # Try extending to next newline
        if next_newline != -1 and next_newline < len(text):
            extended_end = next_newline + 1
            chunk_text = text[start:extended_end]
            tokens = self._count_tokens(chunk_text)
            if tokens <= limit and tokens >= min_chunk_tokens:
                print(f"Extending chunk to next newline at position {extended_end}")
                return extended_end
        
        # Try reducing to previous newline
        if prev_newline > start:
            reduced_end = prev_newline + 1
            chunk_text = text[start:reduced_end]
            tokens = self._count_tokens(chunk_text)
            if tokens <= limit and tokens >= min_chunk_tokens:
                print(f"Reducing chunk to previous newline at position {reduced_end}")
                return reduced_end
        
        # Return original end if adjustments aren't suitable
        return end
    
    def _find_new_chunk_end(self, text: str, start: int, end: int) -> int:
        """Find new chunk end when current chunk exceeds token limit"""
        # Reduce end position to try to fit within token limit
        new_end = end - max(1, (end - start) // 10)  # Reduce by 10% each iteration
        if new_end <= start:
            new_end = start + 1  # Ensure at least one character is included
        
        return new_end
    
    def _extract_headers(self, text: str) -> Headers:
        """Extract markdown headers from text"""
        headers: Headers = {}
        header_regex = re.compile(r'(^|\n)(#{1,6})\s+(.*)', re.MULTILINE)
        
        for match in header_regex.finditer(text):
            level = len(match.group(2))
            content = match.group(3).strip()
            key = f'h{level}'
            if key not in headers:
                headers[key] = []
            headers[key].append(content)
        
        return headers
    
    def _update_current_headers(self, current: Headers, extracted: Headers) -> None:
        """Update current headers with newly extracted headers"""
        for level in range(1, 7):
            key = f'h{level}'
            if key in extracted:
                current[key] = extracted[key]
                self._clear_lower_headers(current, level)
    
    def _clear_lower_headers(self, headers: Headers, level: int) -> None:
        """Clear headers of lower levels than specified"""
        for l in range(level + 1, 7):
            key = f'h{l}'
            if key in headers:
                del headers[key]
    
    def _extract_urls_and_images(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Extract URLs and images from markdown text and replace with placeholders"""
        urls: List[str] = []
        images: List[str] = []
        url_index = 0
        image_index = 0
        
        # Extract images
        def replace_image(match):
            nonlocal image_index
            alt_text, url = match.groups()
            images.append(url)
            result = f'![{alt_text}]({{$img{image_index}}})'
            image_index += 1
            return result
        
        # Extract URLs
        def replace_url(match):
            nonlocal url_index
            link_text, url = match.groups()
            urls.append(url)
            result = f'[{link_text}]({{$url{url_index}}})'
            url_index += 1
            return result
        
        # Process images first
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, text)
        # Then process URLs
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_url, content)
        
        return content, urls, images
