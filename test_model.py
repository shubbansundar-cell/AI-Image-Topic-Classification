
from model import classify_image

# Five sample images
images = [
    "farming.jpg",
    "education.jpg",
    "healthcare.jpg",
    "tourism.jpg",
    "transportation.jpeg"
]

print("\nIMAGE TOPIC CLASSIFICATION RESULTS")
print("====================================")

for image_path in images:

    print("\nTesting:", image_path)

    try:
        predicted_topic, confidence = classify_image(image_path)

        print("Predicted Topic:", predicted_topic)
        print("Confidence:", round(confidence, 2), "%")

    except FileNotFoundError:
        print("Error: Image file not found:", image_path)

    except Exception as e:
        print("Error:", e)

print("\n====================================")
print("All image tests completed!")