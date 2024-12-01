# Define the text prompt (e.g., the original image prompt)
prompt = "High-quality mobile ad banner design for Amul brand with strawberry red accents and ice cream scoops in white bowls."

# Generate text embedding
text_inputs = processor(text=prompt, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embedding = model.get_text_features(**text_inputs)

# Normalize the text embedding
text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)

# Convert text embedding to a list or numpy array for storage
text_embedding = text_embedding.squeeze().cpu().numpy()
