import os
import re
import json
from typing import Dict, List, Optional, Any, TypedDict
import tiktoken
from dataclasses import dataclass

class Headers(TypedDict):
    __annotations__ = {
        'h1': List[str],
        'h2': List[str],
        'h3': List[str],
        'h4': List[str],
        'h5': List[str],
        'h6': List[str]
    }

@dataclass
class IDoc:
    text: str
    metadata: Dict[str, Any]  # Will include: tokens, source, mimeType, name, source_uuid, conversation_uuid, uuid, duration, headers, urls, images, screenshots, chunk_index, total_chunks

class TextService:
    SPECIAL_TOKENS = {
        '<|im_start|>': 100264,
        '<|im_end|>': 100265,
        '<|im_sep|>': 100266
    }

    def __init__(self, model_name: str = 'gpt-4'):
        self.model_name = model_name
        self.tokenizer = None

    def _initialize_tokenizer(self, model: Optional[str] = None) -> None:
        if not self.tokenizer or (model and model != self.model_name):
            self.model_name = model or self.model_name
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def _count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            raise Exception('Tokenizer not initialized')
        formatted_content = self._format_for_tokenization(text)
        tokens = self.tokenizer.encode(formatted_content)
        return len(tokens)

    def _format_for_tokenization(self, text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant<|im_end|>"

    async def split(self, text: str, limit: int, metadata: Optional[Dict[str, Any]] = None) -> List[IDoc]:
        print(f"Starting split process with limit: {limit} tokens")
        self._initialize_tokenizer()
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
                metadata={
                    'tokens': tokens,
                    'headers': dict(current_headers),
                    'urls': urls,
                    'images': images,
                    **(metadata or {})
                }
            ))

            print(f"Chunk processed. New position: {chunk_end}")
            position = chunk_end

        print(f"Split process completed. Total chunks: {len(chunks)}")
        return chunks

    def _get_chunk(self, text: str, start: int, limit: int) -> tuple[str, int]:
        print(f"Getting chunk starting at {start} with limit {limit}")

        # Account for token overhead due to formatting
        overhead = self._count_tokens(self._format_for_tokenization('')) - self._count_tokens('')

        # Initial tentative end position
        end = min(start + int((len(text) - start) * limit / self._count_tokens(text[start:])), len(text))

        # Adjust end to avoid exceeding token limit
        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)

        while tokens + overhead > limit and end > start:
            print(f"Chunk exceeds limit with {tokens + overhead} tokens. Adjusting end position...")
            end = self._find_new_chunk_end(text, start, end)
            chunk_text = text[start:end]
            tokens = self._count_tokens(chunk_text)

        # Adjust chunk end to align with newlines without significantly reducing size
        end = self._adjust_chunk_end(text, start, end, tokens + overhead, limit)

        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)
        print(f"Final chunk end: {end}")
        return chunk_text, end

    def _adjust_chunk_end(self, text: str, start: int, end: int, current_tokens: int, limit: int) -> int:
        min_chunk_tokens = int(limit * 0.8)  # Minimum chunk size is 80% of limit

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
        # Reduce end position to try to fit within token limit
        new_end = end - int((end - start) / 10)  # Reduce by 10% each iteration
        if new_end <= start:
            new_end = start + 1  # Ensure at least one character is included
        return new_end

    def _extract_headers(self, text: str) -> Headers:
        headers: Headers = {}
        header_regex = re.compile(r'(^|\n)(#{1,6})\s+(.*)')

        for match in header_regex.finditer(text):
            level = len(match.group(2))
            content = match.group(3).strip()
            key = f'h{level}'
            if key not in headers:
                headers[key] = []
            headers[key].append(content)

        return headers

    def _update_current_headers(self, current: Headers, extracted: Headers) -> None:
        for level in range(1, 7):
            key = f'h{level}'
            if key in extracted:
                current[key] = extracted[key]
                self._clear_lower_headers(current, level)

    def _clear_lower_headers(self, headers: Headers, level: int) -> None:
        for l in range(level + 1, 7):
            key = f'h{l}'
            if key in headers:
                del headers[key]

    def _extract_urls_and_images(self, text: str) -> tuple[str, List[str], List[str]]:
        urls: List[str] = []
        images: List[str] = []
        url_index = 0
        image_index = 0

        def replace_image(match: re.Match) -> str:
            nonlocal image_index
            alt_text, url = match.groups()
            images.append(url)
            result = f"![{alt_text}]({{{{$img{image_index}}}}})"
            image_index += 1
            return result

        def replace_url(match: re.Match) -> str:
            nonlocal url_index
            link_text, url = match.groups()
            if not url.startswith('{{$img'):
                urls.append(url)
                result = f"[{link_text}]({{{{$url{url_index}}}}})"
                url_index += 1
                return result
            return match.group(0)

        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, text)
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_url, content)

        return content, urls, images

    async def document(self, text: str, model: Optional[str] = None, additional_metadata: Optional[Dict[str, Any]] = None) -> IDoc:
        self._initialize_tokenizer(model)
        tokens = self._count_tokens(text)
        headers = self._extract_headers(text)
        content, urls, images = self._extract_urls_and_images(text)

        return IDoc(
            text=content,
            metadata={
                'tokens': tokens,
                'headers': headers,
                'urls': urls,
                'images': images,
                'screenshots': [],  # Added to match TypeScript
                **(additional_metadata or {})
            }
        )

    def restore_placeholders(self, idoc: IDoc) -> IDoc:
        text = idoc.text
        metadata = idoc.metadata

        # Replace image placeholders with actual URLs using regex
        if 'images' in metadata:
            for index, url in enumerate(metadata['images']):
                pattern = rf'\(\{{{{\$img{index}\}}}}\)'
                text = re.sub(pattern, f'({url})', text)

        # Replace URL placeholders with actual URLs using regex
        if 'urls' in metadata:
            for index, url in enumerate(metadata['urls']):
                pattern = rf'\(\{{{{\$url{index}\}}}}\)'
                text = re.sub(pattern, f'({url})', text)

        return IDoc(text=text, metadata=metadata)
