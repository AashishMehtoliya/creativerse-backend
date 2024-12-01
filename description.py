from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
def read_Image():
    image_path = "/Users/aashishmehtoliya/Downloads/testHack.png"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Display the result
    print("Generated Caption:", caption)
