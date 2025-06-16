import os
import json
import uuid
import mimetypes
import asyncio
import aiofiles
import aiohttp
import magic
import pandas as pd
import re
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from text_service import TextService, IDoc
from audio_service import AudioService
from openai_service import OpenAIService
from web_search_service import WebSearchService

class FileService:
    def __init__(self):
        self.text_service = TextService()
        self.audio_service = AudioService()
        self.openai_service = OpenAIService()
        self.web_search_service = WebSearchService()

        # Constants and Configuration
        self.SCOPES = [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/documents",
        ]

        self.TEMP_DIR = "storage/temp"

        self.MIME_TYPES = {
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xls': 'application/vnd.ms-excel',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'pdf': 'application/pdf',
            'googleDoc': 'application/vnd.google-apps.document',
            'googleSheet': 'application/vnd.google-apps.spreadsheet',
        }

        self.mime_types = {
            'text': {
                'extensions': ['.txt', '.md', '.json', '.html', '.csv'],
                'mimes': [
                    'text/plain',
                    'text/markdown',
                    'application/json',
                    'text/html',
                    'text/csv',
                ],
            },
            'audio': {
                'extensions': ['.mp3', '.wav', '.ogg'],
                'mimes': ['audio/mpeg', 'audio/wav', 'audio/ogg'],
            },
            'image': {
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
                'mimes': [
                    'image/jpeg',
                    'image/png',
                    'image/gif',
                    'image/bmp',
                    'image/webp',
                ],
            },
            'document': {
                'extensions': ['.pdf', '.doc', '.docx', '.xls', '.xlsx'],
                'mimes': [
                    'application/pdf',
                    'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/vnd.ms-excel',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                ],
            },
        }

        self.auth_client = None
        #self.initialize_google_auth()

    def initialize_google_auth(self):
        try:
            credentials = {
                'type': 'service_account',
                'project_id': os.getenv('GOOGLE_PROJECT_ID'),
                'private_key_id': os.getenv('GOOGLE_PRIVATE_KEY_ID'),
                'private_key': os.getenv('GOOGLE_PRIVATE_KEY', '').replace('\\n', '\n'),
                'client_email': os.getenv('GOOGLE_CLIENT_EMAIL'),
                'client_id': os.getenv('GOOGLE_CLIENT_ID'),
                'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
                'client_x509_cert_url': f'https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv("GOOGLE_CLIENT_EMAIL")}',
            }

            self.auth_client = service_account.Credentials.from_service_account_info(
                credentials, scopes=self.SCOPES
            )
        except Exception as error:
            print(f'Failed to initialize Google Auth: {str(error)}')
            raise error

    async def write_temp_file(self, file_content: bytes, file_name: str) -> str:
        temp_uuid = str(uuid.uuid4())
        temp_file_path = os.path.join(self.TEMP_DIR, f'{file_name}-{temp_uuid}')

        try:
            os.makedirs(self.TEMP_DIR, exist_ok=True)
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(file_content)

            # Check MIME type after writing the file
            await self.check_mime_type(temp_file_path, 'text')

            return temp_file_path
        except Exception as error:
            print(f'Failed to write temp file: {str(error)}')
            raise error

    async def save(
        self,
        file_content: bytes,
        file_name: str,
        file_uuid: str,
        type: str,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            date = datetime.now()
            date_path = f'{date.year}-{date.month:02d}-{date.day:02d}'
            dir_path = os.path.join(
                os.path.dirname(__file__),
                f'storage/{type}/{date_path}/{file_uuid}'
            )
            os.makedirs(dir_path, exist_ok=True)

            # Determine the MIME type and extension
            mime_type = await self.get_mime_type_from_buffer(file_content, file_name)
            original_ext = os.path.splitext(file_name)[1][1:]
            file_ext = original_ext or mimetypes.guess_extension(mime_type) or self.get_default_extension(type)

            # Ensure the MIME type matches the expected type
            if mime_type not in self.mime_types[type]['mimes']:
                raise ValueError(f'File MIME type {mime_type} does not match expected type {type}')

            # Construct the new file name
            file_name_without_ext = os.path.splitext(file_name)[0]
            new_file_name = f'{file_name_without_ext}.{file_ext}'
            file_path = os.path.join(dir_path, new_file_name)

            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)

            result = {
                'type': type,
                'path': file_path,
                'file_name': new_file_name,
                'mime_type': mime_type,
                'file_uuid': file_uuid,
            }
            if source:
                result['source'] = source

            print('File saved to:', result)
            return result
        except Exception as error:
            print(f'Failed to save file: {str(error)}')
            raise error

    async def read_text_file(self, original_path: str, storage_path: str) -> IDoc:
        try:
            mime_type = await self.get_mime_type(storage_path)

            if mime_type not in self.mime_types['text']['mimes']:
                raise ValueError(f'Unsupported text file MIME type: {mime_type}')

            async with aiofiles.open(storage_path, 'r', encoding='utf-8') as f:
                text = await f.read()

            additional_metadata = {
                'source': original_path,
                'path': storage_path,
                'name': os.path.basename(original_path),
                'mime_type': mime_type,
            }

            doc = await self.text_service.document(text, None, additional_metadata)
            return doc
        except Exception as error:
            print(f'Failed to read text file: {str(error)}')
            raise error

    async def get_mime_type(self, file_path: str) -> str:
        try:
            if not isinstance(file_path, str):
                raise ValueError('Invalid file path: must be a string')

            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except Exception as error:
            print(f'Failed to get MIME type: {str(error)}')
            raise error

    async def get_mime_type_from_buffer(self, file_buffer: bytes, file_name: str) -> str:
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(file_buffer)
            if mime_type:
                return mime_type

            # Fallback to mimetypes module
            mime_type = mimetypes.guess_type(file_name)[0]
            if mime_type:
                return mime_type

            return 'application/octet-stream'
        except Exception as error:
            print(f'Failed to get MIME type from buffer: {str(error)}')
            raise error

    async def check_mime_type(self, file_path: str, type: str) -> None:
        try:
            mime_type = await self.get_mime_type(file_path)
            if mime_type not in self.mime_types[type]['mimes']:
                raise ValueError(f'Unsupported MIME type for {type}: {mime_type}')
        except Exception as error:
            print(f'Failed to check MIME type: {str(error)}')
            raise error

    def get_default_extension(self, type: str) -> str:
        default_extensions = {
            'audio': 'mp3',
            'text': 'txt',
            'image': 'jpg',
            'document': 'bin'
        }
        return default_extensions.get(type, 'bin')

    async def fetch_and_save_url_file(self, url: str, file_uuid: str) -> Dict[str, Any]:
        try:
            parsed_url = urllib.parse.urlparse(url)
            file_name = os.path.basename(parsed_url.path)

            # Handle query parameters in filename
            if '?' in file_name:
                file_name = file_name.split('?')[0]

            file_extension = os.path.splitext(file_name)[1].lower()

            # If no file extension, try web scraping
            if not file_extension:
                scraped_content = await self.web_search_service.scrape_urls([url], file_uuid)

                if scraped_content and scraped_content[0].content:
                    content = scraped_content[0].content
                    file_name = f"{parsed_url.netloc}_{file_uuid}.md"
                    file_content = content.encode('utf-8')

                    saved_file = await self.save(file_content, file_name, file_uuid, 'text', url)
                    return {**saved_file, 'mime_type': 'text/markdown'}
                else:
                    raise ValueError('Failed to scrape content from the URL')

            # Download file from URL
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f'Failed to fetch file from URL: {response.status}')

                    file_content = await response.read()

                    # Determine MIME type
                    if file_extension in self.mime_types['text']['extensions']:
                        mime_type = mimetypes.guess_type(file_name)[0] or 'text/plain'
                    else:
                        mime_type = response.headers.get('content-type', 'application/octet-stream')

                    file_type = self.get_file_category_from_mime_type(mime_type)

                    if not file_name:
                        file_name = f'file_{file_uuid}{file_extension}'

                    saved_file = await self.save(file_content, file_name, file_uuid, file_type, url)
                    print(f'File fetched and saved: {saved_file["path"]}')

                    return {**saved_file, 'mime_type': mime_type}

        except Exception as error:
            print(f'Failed to fetch and save file from URL: {str(error)}')
            raise error

    def get_file_category_from_mime_type(self, mime_type: str) -> str:
        for category, info in self.mime_types.items():
            if mime_type in info['mimes']:
                return category
        return 'document'  # Default to document if no match is found

    async def process(self, file_path_or_url: str, chunk_size: Optional[int] = None) -> Dict[str, List[IDoc]]:
        try:
            if file_path_or_url.startswith(('http://', 'https://')):
                file_uuid = str(uuid.uuid4())
                file_info = await self.fetch_and_save_url_file(file_path_or_url, file_uuid)
                original_path = file_path_or_url
                storage_path = file_info['path']
            else:
                original_path = os.path.join(os.path.dirname(__file__), file_path_or_url)
                file_uuid = str(uuid.uuid4())
                with open(original_path, 'rb') as f:
                    file_content = f.read()
                mime_type = await self.get_mime_type_from_buffer(file_content, os.path.basename(original_path))
                file_type = self.get_file_category_from_mime_type(mime_type)
                file_info = await self.save(file_content, os.path.basename(original_path), file_uuid, file_type, original_path)
                storage_path = file_info['path']

            docs = []
            screenshot_paths = None

            mime_type = await self.get_mime_type(storage_path)
            file_type = self.get_file_category_from_mime_type(mime_type)

            if file_type == 'audio':
                # Handle audio processing
                chunks = await self.audio_service.split(storage_path, 25)
                transcriptions = await self.openai_service.transcribe(chunks, {'language': 'pl', 'fileName': os.path.splitext(os.path.basename(original_path))[0] + '.md'})

                for index, transcription in enumerate(transcriptions):
                    metadata = {
                        **transcription.metadata,
                        'chunk_index': index,
                        'total_chunks': len(transcriptions),
                        'source_uuid': file_uuid,
                        'uuid': str(uuid.uuid4())
                    }

                    if chunk_size:
                        chunk_docs = await self.text_service.split(transcription.text, chunk_size, metadata)
                        docs.extend(chunk_docs)
                    else:
                        doc = await self.text_service.document(transcription.text, None, metadata)
                        docs.append(doc)

                # Clean up chunks
                for chunk in chunks:
                    os.remove(chunk)

            elif file_type == 'text':
                # Handle text processing
                with open(storage_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()

                base_metadata = {
                    'source': original_path,
                    'path': storage_path,
                    'name': os.path.basename(original_path),
                    'mime_type': mime_type,
                    'source_uuid': file_uuid
                }

                if chunk_size:
                    chunk_docs = await self.text_service.split(text_content, chunk_size, base_metadata)
                    docs = []
                    for doc in chunk_docs:
                        # Ensure metadata stays small by only including essential fields
                        doc.metadata.update({
                            'uuid': str(uuid.uuid4()),
                            'chunk_index': len(docs),
                            'total_chunks': len(chunk_docs)
                        })
                        docs.append(doc)
                else:
                    doc = await self.text_service.document(text_content, None, {
                        **base_metadata,
                        'uuid': str(uuid.uuid4())
                    })
                    docs = [doc]

            elif file_type == 'document':
                # Handle document processing
                doc_content = await self.read_document_file(original_path, storage_path)

                if chunk_size:
                    docs = await self.text_service.split(doc_content.text, chunk_size)
                else:
                    docs = [doc_content]

                # Take screenshots of the document
                screenshot_paths = await self.take_screenshot(storage_path, os.path.basename(original_path))
                print(f'Screenshots saved to: {screenshot_paths}')

                # Add screenshot information to metadata
                if screenshot_paths:
                    for index, doc in enumerate(docs):
                        doc.metadata = {
                            **doc.metadata,
                            'source_uuid': file_uuid,
                            'uuid': str(uuid.uuid4()),
                            'screenshots': screenshot_paths,
                            'chunk_index': index,
                            'total_chunks': len(docs)
                        }

            elif file_type == 'image':
                # Handle image processing
                image_descriptions = await self.openai_service.process_images([storage_path])
                doc = await self.text_service.document(
                    image_descriptions[0].description,
                    None,
                    {
                        'source': original_path,
                        'path': storage_path,
                        'name': os.path.basename(original_path),
                        'mime_type': mime_type,
                        'source_uuid': file_uuid,
                        'uuid': str(uuid.uuid4())
                    }
                )
                docs = [doc]

            else:
                raise ValueError(f'Unsupported file type: {file_type}')

            return {'docs': docs}

        except Exception as error:
            print(f'Failed to process file: {str(error)}')
            raise error

    async def read_document_file(self, original_path: str, storage_path: str) -> IDoc:
        try:
            mime_type = await self.get_mime_type(storage_path)

            if mime_type not in self.mime_types['document']['mimes']:
                raise ValueError(f'Unsupported document file MIME type: {mime_type}')

            content = None

            if mime_type in [
                self.MIME_TYPES['doc'],
                self.MIME_TYPES['docx'],
                self.MIME_TYPES['xls'],
                self.MIME_TYPES['xlsx']
            ]:
                print("Processing office file...", mime_type)
                markdown, _ = await self.process_office_file(storage_path)
                content = markdown
            elif mime_type == self.MIME_TYPES['pdf']:
                content = await self.read_pdf_file(storage_path)
            else:
                raise ValueError(f'Unsupported document file MIME type: {mime_type}')

            additional_metadata = {
                'source': original_path,
                'path': storage_path,
                'name': os.path.basename(original_path),
                'mime_type': mime_type,
            }

            doc = await self.text_service.document(content.strip(), None, additional_metadata)
            return doc
        except Exception as error:
            print(f'Failed to read document file: {str(error)}')
            raise error

    async def process_office_file(self, file_path: str) -> tuple[str, str]:
        ext = os.path.splitext(file_path)[1][1:]
        mime_type = self.MIME_TYPES.get(ext)
        if not mime_type:
            raise ValueError(f'Unsupported file type: {ext}')

        temp_files = []

        try:
            file_id = await self.upload_file_to_drive(file_path, mime_type)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            os.makedirs(self.TEMP_DIR, exist_ok=True)

            intermediate_file_path = os.path.join(
                self.TEMP_DIR,
                f'{base_name}.{"csv" if "xl" in ext else "html"}'
            )
            pdf_path = os.path.join(self.TEMP_DIR, f'{base_name}.pdf')

            temp_files.append(intermediate_file_path)

            await self.get_plain_file_contents_from_drive(file_id, intermediate_file_path, mime_type)
            await self.download_as_pdf(file_id, pdf_path, mime_type)

            if 'xl' in ext:
                async with aiofiles.open(intermediate_file_path, 'r', encoding='utf-8') as f:
                    csv_content = await f.read()
                markdown = self.csv_to_markdown(csv_content)
            else:
                markdown = await self.convert_html_to_markdown(intermediate_file_path)

            return markdown, pdf_path
        except Exception as error:
            print(f'Failed to process Office file: {str(error)}')
            raise error
        finally:
            # Clean up temporary files, but not the PDF
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass

    def csv_to_markdown(self, csv_content: str) -> str:
        lines = csv_content.split('\n')
        header_line = lines[0]
        headers = header_line.split(',')
        markdown_lines = [
            f'| {" | ".join(headers)} |',
            f'| {" | ".join(["---"] * len(headers))} |',
            *[f'| {" | ".join(line.split(","))} |' for line in lines[1:] if line.strip()]
        ]
        return '\n'.join(markdown_lines)

    async def convert_html_to_markdown(self, file_path: str) -> str:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            # TODO: Implement HTML to Markdown conversion
            # For now, return the HTML content as is
            return content
        except Exception as error:
            print(f'Failed to convert HTML to Markdown: {str(error)}')
            raise error

    async def read_pdf_file(self, file_path: str) -> str:
        try:
            # Check if pdftohtml is installed
            try:
                await asyncio.create_subprocess_exec('which', 'pdftohtml')
            except:
                raise ValueError('pdftohtml is not installed or not in PATH')

            temp_html_path = f'{file_path}.html'
            temp_files = [temp_html_path]

            try:
                process = await asyncio.create_subprocess_exec(
                    'pdftohtml', '-s', '-i', '-noframes', file_path, temp_html_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                async with aiofiles.open(temp_html_path, 'r', encoding='utf-8') as f:
                    html_content = await f.read()

                # Remove comments and title
                html_content = re.sub(r'<!--[\s\S]*?-->', '', html_content)
                html_content = re.sub(r'<title>.*?</title>', '', html_content, flags=re.IGNORECASE)

                # TODO: Implement HTML to Markdown conversion
                # For now, return the HTML content as is
                return html_content
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        except Exception as error:
            print(f'Failed to read PDF file: {str(error)}')
            raise error

    async def upload_file_to_drive(self, file_path: str, mime_type: str) -> str:
        try:
            auth = await self.auth_client.get_client()
            drive = build('drive', 'v3', credentials=auth)

            file_name = os.path.basename(file_path)
            if not file_name:
                raise ValueError('Invalid file path: unable to determine file name')

            file_metadata = {'name': file_name}
            media = MediaFileUpload(file_path, mimetype=mime_type)

            file = drive.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            if not file.get('id'):
                raise ValueError('Failed to upload file: no ID returned')

            return file['id']
        except Exception as error:
            print(f'Failed to upload file to Drive: {str(error)}')
            raise error

    async def convert_to_drive_format(self, file_id: str, source_mime_type: str) -> str:
        try:
            auth = await self.auth_client.get_client()
            drive = build('drive', 'v3', credentials=auth)

            target_mime_type = (
                self.MIME_TYPES['googleSheet'] if 'sheet' in source_mime_type
                else self.MIME_TYPES['googleDoc']
            )

            file = drive.files().copy(
                fileId=file_id,
                body={'mimeType': target_mime_type}
            ).execute()

            if not file.get('id'):
                raise ValueError('Failed to copy file: no ID returned')

            return file['id']
        except Exception as error:
            print(f'Failed to convert file in Drive: {str(error)}')
            raise error

    async def get_plain_file_contents_from_drive(self, file_id: str, output_path: str, mime_type: str) -> None:
        try:
            converted_id = await self.convert_to_drive_format(file_id, mime_type)
            access_token = await self.auth_client.get_access_token()

            is_sheet = 'sheet' in mime_type
            export_mime_type = 'text/csv' if is_sheet else 'text/html'
            export_url = f'https://www.googleapis.com/drive/v3/files/{converted_id}/export?mimeType={urllib.parse.quote(export_mime_type)}'

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    export_url,
                    headers={'Authorization': f'Bearer {access_token}'}
                ) as response:
                    if response.status != 200:
                        raise ValueError(f'Failed to export file: {response.status}')

                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())

            await self.delete_drive_file(converted_id)
        except Exception as error:
            print(f'Failed to get file contents from Drive: {str(error)}')
            raise error

    async def download_as_pdf(self, file_id: str, output_path: str, mime_type: str) -> None:
        try:
            converted_id = await self.convert_to_drive_format(file_id, mime_type)
            access_token = await self.auth_client.get_access_token()

            is_sheet = 'sheet' in mime_type
            export_url = f'https://docs.google.com/{"spreadsheets" if is_sheet else "document"}/d/{converted_id}/export?format=pdf'
            if is_sheet:
                export_url += '&portrait=false&size=A4&scale=2'

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    export_url,
                    headers={'Authorization': f'Bearer {access_token}'}
                ) as response:
                    if response.status != 200:
                        raise ValueError(f'Failed to download PDF: {response.status}')

                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())

            await self.delete_drive_file(converted_id)
        except Exception as error:
            print(f'Failed to download PDF from Drive: {str(error)}')
            raise error

    async def delete_drive_file(self, file_id: str) -> None:
        try:
            auth = await self.auth_client.get_client()
            drive = build('drive', 'v3', credentials=auth)
            drive.files().delete(fileId=file_id).execute()
        except Exception as error:
            print(f'Failed to delete file from Drive: {str(error)}')

    async def take_screenshot(self, file_path: str, file_name: str) -> List[str]:
        try:
            extension = os.path.splitext(file_path)[1].lower()
            output_base_name = os.path.splitext(file_name)[0]
            uuid_str = str(uuid.uuid4())
            saved_image_paths = []

            pdf_path = None
            options = {
                'density': 300,
                'save_filename': f'{output_base_name}_{uuid_str}',
                'format': 'jpeg',
                'width': 2480,
                'height': 3508,
                'quality': 100
            }

            if extension in ['.doc', '.docx', '.xls', '.xlsx']:
                _, pdf_path = await self.process_office_file(file_path)
                if extension in ['.xls', '.xlsx']:
                    options.update({'width': 3508, 'height': 2480})
            elif extension == '.pdf':
                pdf_path = file_path
            else:
                raise ValueError(f'Unsupported file type for screenshotting: {extension}')

            os.makedirs(self.TEMP_DIR, exist_ok=True)

            # Get page count
            page_count = await self.get_page_count(pdf_path)

            # Convert each page to image
            for i in range(1, page_count + 1):
                output_path = os.path.join(
                    self.TEMP_DIR,
                    f'{output_base_name}_{i}.jpg'
                )

                # Use pdf2pic to convert page to image
                # Note: This is a placeholder - you'll need to implement the actual conversion
                # using a library like pdf2pic or similar
                # For now, we'll just create an empty file
                with open(output_path, 'wb') as f:
                    f.write(b'')

                # Read the image and save it with proper metadata
                with open(output_path, 'rb') as f:
                    image_buffer = f.read()

                saved_image_info = await self.save(
                    image_buffer,
                    f'{output_base_name}_{i}.jpg',
                    uuid_str,
                    'image'
                )

                saved_image_paths.append(saved_image_info['path'])

                # Clean up temporary files
                os.remove(output_path)

            # Clean up temporary PDF file if it was generated
            if pdf_path and pdf_path.startswith(self.TEMP_DIR):
                os.remove(pdf_path)

            return saved_image_paths
        except Exception as error:
            print(f'Failed to take screenshot: {str(error)}')
            raise error

    async def get_page_count(self, pdf_path: str) -> int:
        try:
            # Check if pdfinfo is installed
            try:
                process = await asyncio.create_subprocess_exec(
                    'which', 'pdfinfo',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            except:
                raise ValueError('pdfinfo is not installed or not in PATH')

            process = await asyncio.create_subprocess_exec(
                'pdfinfo', pdf_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()

            for line in stdout.decode().split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())

            raise ValueError('Could not determine page count')
        except Exception as error:
            print(f'Failed to get page count: {str(error)}')
            raise error

    async def load(self, file_path: str) -> Dict[str, Any]:
        try:
            absolute_path = os.path.join(os.path.dirname(__file__), file_path)
            async with aiofiles.open(absolute_path, 'rb') as f:
                data = await f.read()
            mime_type = await self.get_mime_type_from_buffer(data, os.path.basename(file_path))
            return {'data': data, 'mime_type': mime_type}
        except Exception as error:
            print('Failed to load file:', error)
            raise error

    async def save_docs_to_file(self, docs: List[IDoc], file_name: str) -> str:
        try:
            full_content = ''
            for doc in docs:
                restored_doc = self.text_service.restore_placeholders(doc)
                full_content += restored_doc.text + '\n\n'

            file_uuid = str(uuid.uuid4())
            file_content = full_content.encode('utf-8')
            saved_file = await self.save(file_content, file_name, file_uuid, 'text')

            return saved_file['path']
        except Exception as error:
            print(f'Failed to save docs to file: {str(error)}')
            raise error
