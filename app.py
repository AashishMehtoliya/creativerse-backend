from flask import Flask, request,jsonify
from flask_cors import CORS
from image_gen import image_generator
from handle_image_and_prompt import handle_image_and_prompt

import logging


# Making an isntance of the Flask Class (THis will get passed to app.py)
app = Flask(__name__)

CORS(app)

# Define a sample endpoint
@app.route('/api/hackathon/creative', methods=['POST'])  # Use POST for sending variables
def sample_endpoint1():
    params = request.json
    try:
        # Pass the parameters to the image generator
        data = image_generator(params)
        return data
    except Exception as e:
        logging.error(f"Error in image generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/hackathon/creative/analysis', methods=['GET'])
def sample_endpoint2():
    logging.info('Got User Query Request: %s', request.json)
    data = handle_image_and_prompt("/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/generated_images/my-image.jpeg",
        "/Users/aashishmehtoliya/Documents/code/beepkart/hackathon/creativerse/prompts/banner_creative.txt")
    return data

   


# Run the Flask app
# 127.0.0.1 for localhost
app.run(host = '0.0.0.0',port = 5001,debug=True)
app.run(debug=True)