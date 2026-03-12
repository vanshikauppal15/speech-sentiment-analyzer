import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add FFmpeg path so Whisper can find it
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0.1-essentials_build\bin"

from flask import Flask, render_template, request
import whisper
from transformers import pipeline

app = Flask(__name__)

model = whisper.load_model("base")
sentiment = pipeline("sentiment-analysis")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["audio"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = model.transcribe(filepath)
    text = result["text"]

    sentiment_result = sentiment(text)

    label = sentiment_result[0]["label"]
    score = sentiment_result[0]["score"]

    return render_template(
        "index.html",
        transcription=text,
        sentiment=label,
        confidence=score
    )

if __name__ == "__main__":
    app.run(debug=True)