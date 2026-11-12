


from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from model import classify_image
from text_generator import generate_keywords, generate_paragraph
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Topic Classifier</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            color: #1e293b;
        }

        header {
            background: linear-gradient(135deg, #1e3a8a, #2563eb);
            color: white;
            text-align: center;
            padding: 35px 15px;
        }

        header h1 {
            margin: 0;
            font-size: 32px;
        }

        header p {
            margin-top: 10px;
            font-size: 16px;
        }

        .container {
            max-width: 850px;
            margin: 35px auto;
            padding: 25px;
        }

        .card {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        }

        h2 {
            color: #1e40af;
        }

        .upload-box {
            text-align: center;
            padding: 30px;
            border: 2px dashed #93c5fd;
            border-radius: 12px;
            background: #eff6ff;
        }

        input[type="file"] {
            margin: 20px 0;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 13px 28px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result-image {
            display: block;
            max-width: 100%;
            max-height: 350px;
            margin: 20px auto;
            border-radius: 12px;
        }

        .topic {
            text-align: center;
            background: #dbeafe;
            color: #1e40af;
            padding: 18px;
            border-radius: 10px;
            font-size: 22px;
            font-weight: bold;
        }

        .confidence {
            text-align: center;
            color: #15803d;
            font-size: 18px;
            font-weight: bold;
            margin: 18px;
        }

        .keyword-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 0;
            list-style: none;
        }

        .keyword-list li {
            background: #e0e7ff;
            color: #3730a3;
            padding: 10px 16px;
            border-radius: 20px;
            font-size: 14px;
        }

        .description {
            background: #f8fafc;
            padding: 20px;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            line-height: 1.7;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #64748b;
        }

        @media(max-width: 600px) {
            .container {
                padding: 12px;
            }

            header h1 {
                font-size: 25px;
            }
        }
    </style>
</head>

<body>

<header>
    <h1>AI Image Topic Classifier</h1>
    <p>Image Classification and Text Generation Using CLIP and FLAN-T5</p>
</header>

<div class="container">

    <div class="card">
        <h2>Upload Your Image</h2>

        <div class="upload-box">
            <form method="POST" enctype="multipart/form-data">

                <p>Select an image from your computer</p>

                <input
                    type="file"
                    name="image"
                    accept="image/*"
                    required
                >

                <br>

                <button type="submit">Classify Image</button>

            </form>
        </div>
    </div>

    {% if result %}

    <div class="card">

        <h2>Uploaded Image</h2>

        <img
            src="/uploads/{{ filename }}"
            class="result-image"
            alt="Uploaded image"
        >

        <div class="topic">
            Predicted Topic: {{ result }}
        </div>

        <div class="confidence">
            Confidence: {{ confidence }}%
        </div>

    </div>

    <div class="card">

        <h2>Five Related Keywords</h2>

        <ul class="keyword-list">
            {% for word in keywords %}
                <li>{{ word }}</li>
            {% endfor %}
        </ul>

    </div>

    <div class="card">

        <h2>Profession Description</h2>

        <div class="description">
            {{ paragraph }}
        </div>

    </div>

    {% endif %}

</div>

<footer>
    AI Image Topic Classification Project | CLIP + FLAN-T5
</footer>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    keywords = []
    paragraph = None
    filename = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            result, confidence = classify_image(filepath)

            keywords = generate_keywords(result)

            paragraph = generate_paragraph(result)

            confidence = round(confidence, 2)

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence,
        keywords=keywords,
        paragraph=paragraph,
        filename=filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
