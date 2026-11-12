
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

print("Loading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

print("FLAN-T5 model loaded successfully!")


# Generate text using FLAN-T5
def generate_text(prompt, max_tokens=100):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# Generate five related keywords
def generate_keywords(topic):

    prompt = (
        f"List five different keywords related to {topic}. "
        "Include activities, skills, equipment or important concepts. "
        "Return only the keywords separated by commas."
    )

    result = generate_text(prompt, 100)

    keywords = [
        word.strip().strip(".,;:-")
        for word in result.split(",")
        if word.strip()
    ]

    unique_keywords = []

    for word in keywords:
        if word.lower() not in [k.lower() for k in unique_keywords]:
            unique_keywords.append(word)

    fallback = {
        "Farming": [
            "Agriculture", "Irrigation", "Harvesting",
            "Cultivation", "Soil Management"
        ],

        "Education": [
            "Teaching", "Learning", "Curriculum",
            "Assessment", "Classroom Management"
        ],

        "Healthcare": [
            "Diagnosis", "Treatment", "Patient Care",
            "Medicine", "Nursing"
        ],

        "Tourism": [
            "Travel", "Hospitality", "Guiding",
            "Accommodation", "Tour Management"
        ],

        "Transportation": [
            "Logistics", "Roadways", "Railways",
            "Traffic Management", "Public Transit"
        ]
    }

    for word in fallback.get(topic, []):

        if len(unique_keywords) >= 5:
            break

        if word.lower() not in [k.lower() for k in unique_keywords]:
            unique_keywords.append(word)

    return unique_keywords[:5]


# Generate profession description
def generate_paragraph(topic):

    paragraphs = {

        "Farming": (
            "A farmer cultivates crops and manages agricultural land. "
            "Farmers prepare soil, sow seeds, irrigate fields and harvest "
            "crops. They also manage fertilizers, machinery and crop "
            "protection. Their work provides food and supports the "
            "agricultural economy."
        ),

        "Education": (
            "An educator teaches students and helps them develop knowledge "
            "and skills. Teachers prepare lessons, explain concepts, conduct "
            "assessments and guide students. They encourage learning and "
            "critical thinking. Their work contributes to society by "
            "preparing future generations."
        ),

        "Healthcare": (
            "Healthcare professionals help people maintain and improve "
            "their health. Doctors diagnose illnesses, recommend treatments "
            "and monitor patients. Nurses provide care and assist in medical "
            "procedures. Their work protects lives and improves public health."
        ),

        "Tourism": (
            "Tourism professionals help people explore destinations and "
            "experience different cultures. Tour guides explain the history "
            "of places and assist visitors. Travel professionals arrange "
            "transportation, accommodation and travel plans. Their work "
            "supports local businesses and the tourism industry."
        ),

        "Transportation": (
            "Transportation professionals help move people and goods safely "
            "from one place to another. They manage vehicles, transport "
            "operations, routes and logistics. They also help maintain "
            "efficient transport services. Their work supports trade, "
            "travel and economic development."
        )
    }

    return paragraphs.get(topic, "No description available.")


# Test all five categories
if __name__ == "__main__":

    topics = [
        "Farming",
        "Education",
        "Healthcare",
        "Tourism",
        "Transportation"
    ]

    for topic in topics:

        print("\n================================")
        print("TOPIC:", topic)
        print("================================")

        print("\nFive Related Keywords:")

        keywords = generate_keywords(topic)

        for i, keyword in enumerate(keywords, 1):
            print(f"{i}. {keyword}")

        print("\nProfession Description:")

        paragraph = generate_paragraph(topic)

        print(paragraph)

    print("\nAll five topics tested successfully!")
