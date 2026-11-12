
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load pretrained CLIP model
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP model loaded successfully!")


# Define image topics
TOPICS = [
    "Farming",
    "Healthcare",
    "Education",
    "Tourism",
    "Transportation",
    "Food",
    "Animals",
    "Travel"
]


def classify_image(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Prepare image and topic descriptions
    descriptions = [
        "A photograph related to " + topic
        for topic in TOPICS
    ]

    inputs = processor(
        text=descriptions,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Generate predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate similarity probabilities
    probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    # Find highest probability
    best_index = probabilities.argmax().item()

    predicted_topic = TOPICS[best_index]
    confidence = probabilities[best_index].item() * 100

    return predicted_topic, confidence
