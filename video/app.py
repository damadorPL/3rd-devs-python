import os
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_AI_STUDIO_API_KEY not found in environment variables")

# Configure the API
genai.configure(api_key=API_KEY)

class MediaProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_file_size = 3 * 1024 * 1024  # 3MB in bytes
    
    def upload_media_file(self, file_path: str, mime_type: str, display_name: str):
        """Upload media file to Google AI File Manager"""
        file_path = Path(file_path)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File size exceeds 3MB limit. Current size: {file_size / 1024 / 1024:.2f}MB")
            
        print("⏱️ Uploading file...")
        start_time = time.time()
        
        # Upload file
        uploaded_file = genai.upload_file(
            path=str(file_path),
            display_name=display_name
        )
        
        upload_time = time.time() - start_time
        print(f"✅ Upload completed in {upload_time:.2f}s")
        
        return uploaded_file
    
    def wait_for_processing(self, file_name: str):
        """Wait for file processing to complete"""
        print("⏱️ Processing file...")
        start_time = time.time()
        
        # Get file and wait for processing
        file = genai.get_file(file_name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(file_name)
        
        print()  # New line
        processing_time = time.time() - start_time
        print(f"✅ Processing completed in {processing_time:.2f}s")
        
        if file.state.name == "FAILED":
            raise RuntimeError("Media processing failed")
            
        return file
    
    def generate_content(self, processed_file, mime_type: str) -> str:
        """Generate content based on media type"""
        print("⏱️ Generating content...")
        start_time = time.time()
        
        # Choose prompt based on media type
        if mime_type.startswith("video"):
            prompt = "Analyze the content of this video."
        else:
            prompt = "Transcribe and summarize this audio. Write transcription and nothing else."
        
        # Generate content
        response = self.model.generate_content([
            prompt,
            processed_file 
        ])
        
        generation_time = time.time() - start_time
        print(f"✅ Content generated in {generation_time:.2f}s")
        
        return response.text
    
    def delete_uploaded_file(self, file_name: str):
        """Delete uploaded file from Google AI service"""
        print("⏱️ Deleting file...")
        start_time = time.time()
        
        try:
            genai.delete_file(file_name)
            delete_time = time.time() - start_time
            print(f"✅ File deleted in {delete_time:.2f}s: {file_name}")
        except Exception as e:
            print(f"❌ Failed to delete file {file_name}: {e}")
    
    def process_media(self, file_path: str, mime_type: str, display_name: str):
        """Complete media processing workflow"""
        try:
            # Upload the media file
            uploaded_file = self.upload_media_file(file_path, mime_type, display_name)
            
            # Wait for processing
            processed_file = self.wait_for_processing(uploaded_file.name)
            
            print(f"📁 Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")
            
            # Generate content
            content = self.generate_content(processed_file, processed_file.mime_type)
            print("\n" + "="*50)
            print("GENERATED CONTENT:")
            print("="*50)
            print(content)
            print("="*50)
            
            # Clean up
            self.delete_uploaded_file(uploaded_file.name)
            
        except Exception as e:
            print(f"❌ Error processing media: {e}")
            raise

def main():
    """Main execution function"""
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Configuration
    media_config = {
        'path': current_dir / 'test.mp3',  # Change to 'test.mp4' for video
        'mime_type': 'audio/mp3',  # Change to 'video/mp4' for video  
        'display_name': 'Test Audio'  # Change accordingly
    }
    
    # Process media
    processor = MediaProcessor()
    processor.process_media(
        media_config['path'], 
        media_config['mime_type'], 
        media_config['display_name']
    )

if __name__ == "__main__":
    main()
