import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from generate_embedding import generate_image_and_text_embeddings
from utility import validate_image
from image_gen import read_prompt
import boto3
import base64
import json



runtime = session.client("bedrock-runtime","us-west-2")



template_prompt = """Analyze the following image generation prompt and extract key details to create a structured JSON object. The JSON should include expected values for design specifications, text content, colors, dimensions, and constraints. Use the following guidelines:

### Input Prompt:
<Insert Prompt Text Here>

### Output JSON Structure:
{
    "expected_values": {
        "width": <Extracted Width>,
        "height": <Extracted Height>,
        "brand_name": "<Extracted Brand Name>",
        "tagline": "<Extracted Tagline>",
        "offer": "<Extracted Offer Text>",
        "primary_color_hex": "<Extracted Primary Color in HEX>",
        "accent_color_hex": "<Extracted Accent Color in HEX>",
        "product_description": "<Extracted Product Description>",
        "aesthetic_tone": "<Extracted Aesthetic Tone>",
        "contrast_level": "<Extracted Contrast Level>",
        "font_style": "<Extracted Font Style>",
        "brand_name_font_size": <Extracted Brand Name Font Size>,
        "tagline_font_size": <Extracted Tagline Font Size>,
        "product_labels": ["<Relevant Product Label 1>", "<Relevant Product Label 2>", ...],
        "image_constraints": {
            "exact_dimensions": <true/false>,
            "width_constraint": <Extracted Width>,
            "height_constraint": <Extracted Height>,
            "no_alternative_dimensions": <true/false>
        }
    }
}
### Notes:
- Ensure all color codes are in HEX format.
- Validate dimensions and ensure they are precise if mentioned.
- For missing or ambiguous details, infer logical defaults or leave them as placeholders (e.g., "<Extracted Value>").
- Use descriptive and logical keys for clarity and consistency."""

def handle_image_and_prompt(image_path, prompt):

    final_prompt = template_prompt.replace("<Insert Prompt Text Here>", read_prompt("/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/prompts/sample_prompt.txt"))

    body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                        ],
                    }
                ],
            }
        )

    response = runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )

    response_body = json.loads(response.get("body").read())

    cleaned_response = response_body["content"][0]["text"].replace('```json', '').replace('```', '').strip()

    results = validate_image(image_path,  cleaned_response)

    json_string = json.dumps(results)

    print(json_string)  # This will print the dictionary as a JSON string

    data_from_string = json.loads(json_string)
    return data_from_string    

# no use
def evaluate_match(image_embedding, text_embedding):
    # Compute cosine similarity between the image and text embeddings
    cosine_sim = cosine_similarity([image_embedding], [text_embedding])[0][0]

    # You can also compute Euclidean distance if you prefer that
    euclidean_dist = np.linalg.norm(image_embedding - text_embedding)
    
    # Threshold for considering a "good match"
    similarity_threshold = 0.8
    distance_threshold = 0.5  # Adjust based on the scale of your embeddings

    if cosine_sim >= similarity_threshold:
        print(f"Good Match! Cosine Similarity: {cosine_sim}")
    else:
        print(f"Poor Match. Cosine Similarity: {cosine_sim}")

    if euclidean_dist <= distance_threshold:
        print(f"Good Match! Euclidean Distance: {euclidean_dist}")
    else:
        print(f"Poor Match. Euclidean Distance: {euclidean_dist}")
    
    # Return the similarity score for further analysis or use
    return cosine_sim, euclidean_dist