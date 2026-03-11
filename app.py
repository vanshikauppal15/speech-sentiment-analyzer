from flask import Flask, render_template, request
import whisper
from transformers import pipeline
import os

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

    return f"""
    <h2>Transcription</h2>
    <p>{text}</p>

    <h2>Sentiment</h2>
    <p>{label} (confidence {score:.2f})</p>

    <br><a href="/">Try Another Audio</a>
    """

if __name__ == "__main__":
    app.run(debug=True)