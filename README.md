Problem Statement
With the rapid growth of artificial intelligence, image processing and natural language processing have become
important technologies for understanding and interpreting digital information. The objective of this project is to
develop an AI-based Image-to-Topic Classification System that can automatically identify the subject of an
uploaded image and generate relevant textual information based on the identified topic
The system focuses on five major categories: Farming, Education, Healthcare, Tourism and Transportation.
When a user uploads an image, CLIP identifies the most relevant category. FLAN-T5 then generates five related
keywords, while a predefined description provides a short explanation of the corresponding profession. Python
and Flask are used to deliver the functionality through a browser

Motivation
Images contain valuable information about activities, professions, locations and sectors. Manual identification
and description may be repetitive and time-consuming. A combined image-and-text system can demonstrate a
more convenient workflow: visual input is converted into a topic label, and the label is enriched with language
output Another motivation is to study pretrained AI models in a practical setting. Rather than undertaking expensive
model training, the project uses publicly available checkpoints and concentrates on inference, integration,
interface design and evaluation. This approach is appropriate for a student prototype where computational
resources and labelled data may be limited.


PROPOSED METHODOLOGY

The methodology follows a sequential workflow. Each component has a defined input and output so that the
complete system can be tested independently and as an integrated application.

Processing Pipeline
User Image Upload → Image Preprocessing → CLIP Model → Topic Classification → FLAN-T5 → Keyword
Generation → Profession Description → Display Results

Stage 1: Image Upload
The user selects an image using the file input control. The Flask route receives the submitted file and checks
whether a file has been provided. The uploaded image is stored in the application&#39;s upload directory for
processing and display. A production deployment should additionally validate file type, file size and file content

Stage 2: Image Preprocessing
The image is opened using Pillow (PIL) and converted to RGB. This ensures that grayscale, palette or other
supported image modes are normalized before being passed to the model processor. The CLIP processor
performs the model-specific transformations, including resizing, cropping, normalization and tensor preparation
as required by the checkpoint

Stage 3: CLIP Classification
CLIP compares the uploaded image with textual prompts representing the five categories. The image encoder
produces a visual embedding and the text encoder produces embeddings for the candidate descriptions.
Similarity is calculated between the image embedding and each text embedding. The highest scoring category is
selected as the predicted topic. The implementation applies softmax to the similarity logits to obtain normalized values that sum to one across the candidate classes. The largest normalized value is displayed as a confidence-like score. This value indicates relative preference among the supplied candidate labels; it is not equivalent to empirical accuracy or a calibrated probability of correctness.

Stage 4: FLAN-T5 Keyword Generation
The predicted topic is passed to FLAN-T5 through an instruction prompt asking for five relevant keywords. The
tokenizer converts the prompt into model inputs, and the encoder-decoder model generates output tokens.
Decoding converts those tokens into readable text. The application then separates the generated response into
candidate keywords, removes duplicates and applies a fallback list if fewer than five valid entries are available.

Stage 5: Profession Description
The current implementation uses a predefined dictionary to map each topic to a short profession description.
This provides consistent, predictable text and avoids the variability of generated paragraphs. The paragraph is
category-level information and should not be treated as an exhaustive occupational description.

Stage 6: Results Display
Flask passes the uploaded image path, predicted topic, score, keywords and description to the results template.
HTML structures the content and CSS styles the cards, labels and spacing. The result is rendered in the browser
without requiring the user to understand model internals.

MODELS AND TOOLS USED
The application combines pretrained deep learning models with conventional software components. This chapter
describes the role of each technology and the reason for its selection

CLIP: Contrastive Language–Image Pre-training
CLIP was introduced by OpenAI and trained on image-text pairs to learn transferable visual representations. It
contains an image encoder and a text encoder. During pretraining, representations of matching image-text pairs
are encouraged to be close, while non-matching pairs are separated. This alignment enables the model to
compare images with natural-language prompts
The project uses the checkpoint openai/clip-vit-base-patch32. ViT-B/32 denotes a Vision Transformer backbone
with a base configuration and patch size of 32 pixels. The checkpoint is used in inference mode and is not fine-
tuned on a project-specific dataset

Zero-Shot Classification
Zero-shot classification refers to predicting a label without training a dedicated classifier on labelled examples
from the target categories. The candidate categories are supplied as text. CLIP&#39;s pretrained image-text alignment
makes it possible to rank these descriptions by similarity. In this project, the candidate list is fixed to five broad
topics

Candidate Prompt Design
Prompt wording influences the similarity score. A category can be represented by a simple label or a descriptive
sentence such as “a photograph related to farming and agricultural activity”. Consistent prompt structure reduces
accidental differences between candidate descriptions. Prompt design should be documented and kept fixed
during comparative evaluation

FLAN-T5 Base
FLAN-T5 is an instruction-tuned variant of the T5 text-to-text architecture. T5 treats tasks as text input and text
output. FLAN instruction tuning improves the model&#39;s ability to follow natural-language task descriptions. The
project uses google/flan-t5-base to generate topic-related keywords

The model receives an instruction containing the predicted topic and a request for five concise keywords.
Generation parameters such as maximum output length and decoding strategy affect the result. The output is not
guaranteed to be perfectly structured; therefore, the application performs basic parsing and fallback handling

Python and PyTorch
Python provides the application logic and integration layer. PyTorch executes tensor operations and neural-
network inference. The Transformers library supplies pretrained model classes, tokenizers and processors.
Pillow supports image loading and RGB conversion

Flask, HTML and CSS
Flask is a lightweight web framework used to define routes, receive form submissions and render templates.
HTML provides the structure of the upload and results pages. CSS controls layout, colors, spacing, typography
and responsive behavior. Werkzeug utilities support safe filename handling


RESULTS AND SAMPLE OUTPUTS
This chapter presents the application outputs observed during testing. The screenshots show the topic selected
by the model and the accompanying score, keywords and profession description

Summary of Demonstration Results
Test image Predicted topic Displayed score Classroom Education    99.3%
Agricultural field Farming                                        95.07%
Hospital scene Healthcare                                         99.39%
Taj Mahal / visitors Tourism                                      63.77%
City bus Transportation                                           84.87%

Education Result
The classroom image contains a teacher standing near the board and students seated at desks. CLIP ranks the
education-related prompt highest. The output keywords include education, equipment, teaching, learning and
curriculum. The profession description explains lesson preparation, concept explanation, assessment and
guidance.

Farming Result
The farming image shows a person working in a cultivated field. The predicted category is Farming. Keywords
include farm, equipment, farming, agriculture and irrigation. The description summarizes crop cultivation, soil
preparation, sowing, irrigation, harvesting and machinery management

Healthcare Result
The healthcare image shows a medical professional beside a patient in a hospital setting. The predicted topic is
Healthcare. Related keywords include healthcare, equipment, diagnosis, treatment and patient care. The
profession paragraph refers to diagnosis, treatment, monitoring and nursing support

Tourism Result
The tourism image depicts visitors at a prominent destination. CLIP selects Tourism, with a displayed score of
63.77%. The relatively lower score among the sample outputs illustrates that the normalized score can vary
according to image content and competition between candidate prompts. Keywords include tourism, equipment,
travel, hospitality and guiding

Transportation Result
The transportation image shows a public bus on a road. The predicted category is Transportation and the
displayed score is 84.87%. Keywords include transportation, equipment, logistics, roadways and railways. The
description refers to moving people and goods, vehicles, routes, logistics and efficient transport services

Keyword and Profession Description Panels
The keyword panel presents five short terms as individual visual tags. The profession-description panel provides
a paragraph in a bordered text area. These two panels enrich the category label: keywords provide quick
scanning, while the paragraph supplies context. They are generated or selected after classification and do not
influence the CLIP prediction itself
