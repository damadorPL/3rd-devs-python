import replicate
import requests
from PIL import Image
import os
from dotenv import load_dotenv

# Load api keys from .env file
load_dotenv()

# Verify the API token is loaded
if not os.getenv("REPLICATE_API_TOKEN"):
    raise ValueError("REPLICATE_API_TOKEN not found in environment variables. Please check your .env file.")

# Input parameters
input_data = {
    "image": "https://cloud.overment.com/Xnapper-2024-09-24-18.05.21-1727193974.png",
    "query": "Alice app",
    "box_threshold": 0.2,
    "text_threshold": 0.2
}

def download_image(url, file_path):
    """Download image from URL and save to file"""
    response = requests.get(url)
    response.raise_for_status()
    
    with open(file_path, 'wb') as f:
        f.write(response.content)

def process_image():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    input_image_path = os.path.join(script_dir, 'screenshot.png')
    output_image_path = os.path.join(script_dir, 'screenshot_cropped.png')
    segmented_image_path = os.path.join(script_dir, 'screenshot_segmented.png')
    
    # Run the Grounding DINO model
    print("Running AI model for object detection...")
    output = replicate.run(
        "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
        input=input_data
    )
    
    # Download images
    print("Downloading original image...")
    download_image(input_data["image"], input_image_path)
    
    print("Downloading segmented image...")
    download_image(output["result_image"], segmented_image_path)
    
    # Check if input file exists
    if not os.path.exists(input_image_path):
        print(f'Error: Input file is missing: {input_image_path}')
        return
    
    # Find the bounding box with the highest confidence
    highest_confidence_detection = max(output["detections"], key=lambda x: x["confidence"])
    bounding_box = highest_confidence_detection["bbox"]
    
    print(f"Highest confidence detection: {highest_confidence_detection['confidence']:.3f}")
    
    # Calculate crop dimensions
    left = int(bounding_box[0])
    top = int(bounding_box[1])
    right = int(bounding_box[2])
    bottom = int(bounding_box[3])
    
    # Crop the image using PIL
    try:
        with Image.open(input_image_path) as img:
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(output_image_path)
        
        print(f'Image cropped and saved as {output_image_path}')
    
    except Exception as err:
        print(f'Error cropping image: {err}')

if __name__ == "__main__":
    process_image()
