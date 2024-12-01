import imp
import cv2
import torch
# import pytesseract
from PIL import Image
from colorthief import ColorThief
from transformers import CLIPProcessor, CLIPModel
from spellchecker import SpellChecker
import numpy as np
import json
import easyocr
import boto3
import math

# Create an OCR reader object
rekognition = session.client(service_name =  'rekognition',
        region_name = "us-west-2")

# 1. Check image dimensions
def check_image_dimensions(image_path, required_width=320, required_height=704):
    image = Image.open(image_path)
    width, height = image.size
    print("width : "+str(width) + " height : "+str(height))
    diff = abs(width-required_width) + abs(height-required_height)
    return bool(diff < 20)

# 2. Check image orientation (portrait)
def check_image_orientation(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return bool(height > width)

# 3. Check color scheme
def euclidean_distance(color1, color2):
    """Calculate the Euclidean distance between two RGB colors."""
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def hex_to_rgb(hex_value):
    """Convert a hex color value to RGB tuple."""
    print("hex value :")
    print(hex_value)
    hex_value = hex_value.lstrip('#').strip()  # Remove '#' and any surrounding spaces
    if len(hex_value) != 6:
        raise ValueError(f"Invalid hex color value: {hex_value}")
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

def check_color_scheme(image_path, expected_values):
    print(expected_values)
    target_colors = hex_to_rgb(expected_values["primary_color_hex"])
    dominant_color = hex_to_rgb(expected_values["accent_color_hex"])
    threshold = 50  # Distance threshold for acceptable color match
    
    # Get the dominant color of the image using ColorThief
    color_thief = ColorThief(image_path)
    dominant_color_from_image = color_thief.get_color(quality=1)
    
    # Print for debugging
    print(f"Target Colors: {target_colors}")
    print(f"Dominant Color from Image: {dominant_color_from_image}")
    
    # Check proximity of dominant color to each target color
    distance = euclidean_distance(dominant_color_from_image, target_colors)
    print(f"Distance to target color {dominant_color_from_image}: {distance}")
        
    # If the distance is within the threshold, return True
    if distance < threshold:
        print(f"Color {dominant_color_from_image} is within acceptable proximity to {dominant_color_from_image}")
        return bool(True)
    
    # If no color is close enough, return False
    print(f"Color {dominant_color_from_image} is not close enough to any target colors.")
    return bool(False)

def check_similarity(detected_objects, product_labels):
    # Compare each product label with detected objects
    count = 0
    for label in product_labels:
        if label.lower() in [detected_object.lower() for detected_object in detected_objects]:
            count = count+1
    return bool(product_labels-count==0)

# 4. Extract text from image using OCR (Tesseract)
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])

    # Read text from an image
    result = reader.readtext(image_path)

    # Print the extracted text
    for detection in result:
        return detection[1]
    # image = Image.open(image_path)
    # text = pytesseract.image_to_string(image)
    # return text.strip()

# 5. Check text content in extracted OCR text
def check_text_content(extracted_text, brand_name, tagline, offer):
    return bool(brand_name in extracted_text and tagline in extracted_text and offer in extracted_text)


def check_product_in_image():
    bucket_name = "artimagebucket-633968681041"
    object_name= "my-image.jpeg"
    response = rekognition.detect_labels(
        Image={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': object_name
            }
        },
        MaxLabels=10,  # Adjust number of labels you want
        MinConfidence=70  # Confidence threshold
    )

    labels = response['Labels']
    detected_objects = [label['Name'] for label in labels]
    print(detected_objects)
    return detected_objects

# 6. Check product in image using CLIP model
def check_labels_similarity(product_labels, expected_labels):
    product_labels = " ".join(product_labels)
    expected_labels = " ".join(expected_labels)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Process the product labels and get its features
    text_inputs = processor(text=product_labels, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        product_label_features = model.get_text_features(**text_inputs)

    # Process the expected labels and get their features
    expected_text_inputs = processor(text=expected_labels, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        expected_label_features = model.get_text_features(**expected_text_inputs)

    # Calculate cosine similarity between product labels and expected labels
    similarities = torch.cosine_similarity(product_label_features, expected_label_features)

    # Print out the similarities
    print(f"Similarities between product labels and expected labels: {similarities}")
    
    # Get the max similarity score
    max_similarity = similarities.max().item()

    # Print the max similarity score
    print(f"Max similarity: {max_similarity}")

    # If the highest similarity is above a threshold (e.g., 0.7), consider it a match
    return bool(max_similarity > 0.65)  # Return True if similarity is above 0.7

def compare_labels(aws_labels, custom_labels):
    print(aws_labels)
    # Extract label names from AWS labels
    aws_label_names = [label['Name'].lower() for label in aws_labels]
    
    # Convert custom labels to lowercase
    custom_labels = [label.lower() for label in custom_labels]
    
    # Matching results
    matches = []
    unmatched_aws = []
    unmatched_custom = []
    
    # Check direct matches
    for aws_label in aws_label_names:
        matching_custom = [custom for custom in custom_labels if custom in aws_label or aws_label in custom]
        
        if matching_custom:
            matches.append({
                'aws_label': aws_label,
                'custom_label': matching_custom[0],
                'match_type': 'direct'
            })
        else:
            unmatched_aws.append(aws_label)
    
    # Check unmatched custom labels
    for custom_label in custom_labels:
        if custom_label not in [match['custom_label'] for match in matches]:
            unmatched_custom.append(custom_label)
    
    # Semantic similarity using difflib
    from difflib import SequenceMatcher
    semantic_matches = []
    
    for aws_label in aws_label_names:
        for custom_label in custom_labels:
            similarity = SequenceMatcher(None, aws_label, custom_label).ratio()
            if similarity > 0.6:  # Threshold for semantic similarity
                semantic_matches.append({
                    'aws_label': aws_label,
                    'custom_label': custom_label,
                    'similarity_score': similarity
                })
    
    return {
        'direct_matches': matches,
        'unmatched_aws_labels': unmatched_aws,
        'unmatched_custom_labels': unmatched_custom,
        'semantic_matches': semantic_matches
    }

# 7. Check image sharpness (Laplacian variance method)
def check_image_sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(laplacian_var > 100)  # Threshold for sharpness

# 8. Check spelling errors in extracted text using SpellChecker
def check_spelling(text):
    spell = SpellChecker()
    misspelled = spell.unknown(text.split())
    return bool(len(misspelled) == 0)  # No misspelled words

# Function to perform all checks and validations
def validate_image(image_path, expected_values):
    # Perform checks
    results = {}
    if isinstance(expected_values, str):
        expected_values = json.loads(expected_values)

    # Ensure expected_values contains the required structure
    if "expected_values" not in expected_values:
        raise ValueError("Missing 'expected_values' key in input data.")
    expected_values = expected_values["expected_values"]

    # 1. Dimension and Orientation Check
    results["dimensions"] = check_image_dimensions(image_path, expected_values["width"], expected_values["height"])
    results["orientation"] = check_image_orientation(image_path)

    # 2. Color Scheme Check
    target_colors = [(255, 182, 193), (255, 215, 0)]  # Example: soft pink and gold in RGB
    results["color_scheme"] = check_color_scheme(image_path, expected_values)

    # 3. OCR and Text Content Check
    extracted_text = extract_text_from_image(image_path)
    results["text_correct"] = check_text_content(extracted_text, expected_values["brand_name"], expected_values["tagline"], expected_values["offer"])

    # 4. Product Validation (CLIP Model)
    expected_labels = expected_values["product_labels"]
    product_labels = check_product_in_image()
    results["product_valid"] = check_labels_similarity(product_labels, expected_labels)

    # 5. Image Quality (Sharpness Check)
    results["image_quality"] = check_image_sharpness(image_path)

    # 6. Spelling Check in extracted text
    results["spelling_check"] = check_spelling(extracted_text)
    return results
