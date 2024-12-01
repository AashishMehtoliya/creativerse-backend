import torch
from PIL import Image
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from image_gen import read_prompt
from transformers import pipeline

def generate_image_and_text_embeddings(image_path, prompt):
    """
    Generate embeddings for an image and a text prompt using the CLIP model.
    
    Parameters:
    - image_path (str): Path to the image file.
    - prompt (str): Text description (prompt).
    
    Returns:
    - tuple: A tuple containing the image and text embeddings as numpy arrays.
    """
    promptString = read_prompt(prompt)
    if len(promptString.split()) > 77:  # Approximation based on number of words
        promptString = summarize_text(promptString)
    # Load pre-trained CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Generate image embedding
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)

    # Generate text embedding
    text_inputs = processor(text=promptString, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)

    # Check the size of the embeddings
    print(f"Image embedding size: {image_embedding.size()}")
    print(f"Text embedding size: {text_embedding.size()}")

    # Define the projection layer to match the dimensions
    projection_dim = 512  # Common projection size, you can choose this depending on the model
    projection_layer = nn.Linear(image_embedding.size(-1), projection_dim)

    # Project the image embedding to the same dimension as text embedding
    image_embedding_projected = projection_layer(image_embedding)

    # Normalize both embeddings
    image_embedding_projected /= image_embedding_projected.norm(p=2, dim=-1, keepdim=True)
    text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)

    image_embedding = image_embedding.detach().cpu().numpy()
    text_embedding = text_embedding.detach().cpu().numpy()

    # Return the projected image and text embeddings as numpy arrays
    return image_embedding.squeeze(), text_embedding.squeeze()


def summarize_text(prompt):
    # Initialize the summarization pipeline (using BART or T5 model)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(prompt, max_length=77, min_length=40, do_sample=False)
    return summary[0]['summary_text']
