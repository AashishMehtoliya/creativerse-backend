import boto3
from flask import send_file
import json
import io, base64
from PIL import Image
import os

                    
s3 = session.client(service_name = 's3',
    region_name = "us-west-2")

#session.client
bedrock = session.client(
    service_name = 'bedrock-runtime',
    region_name = "us-west-2"
    )

def read_prompt(file_path):
    """
    Reads the image prompt from a text file.
    :param file_path: Path to the text file containing the prompt.
    :return: The content of the text file as a string.
    """
    try:
        with open(file_path, 'r') as file:
            prompt = file.read().strip()  # Strip to remove any extra spaces or newlines
        return prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")
        
def image_generator(params):

    variables = params.get("variables")
    if not isinstance(variables, dict):
        raise ValueError("Expected 'variables' to be a dictionary.")
    
    question = read_prompt("/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/prompts/banner_creative.txt")

    
    # Ensure the prompt is a string
    if not isinstance(question, str):
        raise ValueError("Prompt must be a string.")


    if question and variables:
        prompt = question.format(**variables)
    else:
        raise ValueError("Prompt template or variables missing in request.")

    print(prompt)
    try:
        with open("/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/prompts/sample_prompt.txt", "w") as file:
            file.write(prompt)
    except OSError as e:
        raise RuntimeError(f"Error saving the prompt to a file: {e}")
    # The parameters passed to the invocation of the Bedrock Model
    body = json.dumps({
    "prompt": prompt,  # Required: Your image description as a string.
    "seed": 0,  # Optional: Seed for deterministic generation.
    "negative_prompt": "blurry, low quality,No spelling errors, No additional text except brand and tagline, No blurry or distorted text,  No incorrect brand name,No incorrect tagline",  # Optional: To avoid undesired characteristics.
    "output_format": "png"  # Optional: Set the output format (PNG for higher quality).
    })

    # Specifying the specific model we want to use with Amazon Bedrock
    modelId = 'stability.stable-image-ultra-v1:0'

    # Invoke the Bedrock model
    response = bedrock.invoke_model(modelId=modelId, body=body)
    # response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    if 'images' not in response_body:
        raise ValueError(f"Unexpected response format: {response_body}")
    base64_str = response_body['images'][0]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    desired_width = variables.get("WIDTH")
    desired_height = variables.get("HEIGHT")
    # img = img.resize((desired_width, desired_height), Image.Resampling.LANCZOS)
    output_directory = "/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/generated_images"
    output_directory2 = "/Users/aashishmehtoliya/Downloads/hack1/src/assets/generated-images"
    output_path = os.path.join(output_directory, 'my-image.jpeg')
    output_path2 = os.path.join(output_directory2, 'my-image.jpeg')
    
    try:
        img.save(output_path)
        img.save(output_path2)  
        upload_image_to_s3(output_path)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        print(f"Image saved successfully at: {output_path}")
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=False)
    except Exception as e:  
        print(f"Error: {e}")
        raise

def upload_image_to_s3(image_path):
    bucket_name = "artimagebucket-633968681041"
    object_name= "my-image.jpeg"
    s3.upload_file(image_path, bucket_name, object_name)
