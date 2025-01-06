from flask import Flask, request, render_template, jsonify
from services.youtube_service import YouTubeService
from services.rag_service import RAGService
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
youtube_service = YouTubeService()
rag_service = RAGService(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit_video", methods=["POST"])
def submit_video():
    youtube_url = request.form["youtube_url"]
    try:
        video_id = youtube_service.extract_video_id(youtube_url)
        transcript = youtube_service.fetch_transcript(video_id)
        rag_service.process_transcript(transcript, video_id)
        return jsonify({"status": "success", "message": "Video transcript stored successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/ask_question", methods=["POST"])
def ask_question():
    question = request.form["question"]
    video_id = request.form.get("video_id", "")
    try:
        retriever = rag_service.get_retriever(video_id)
        if not retriever:
            return jsonify({"status": "error", "message": "No video transcript stored. Please submit a video first."}), 400
        answer = rag_service.answer_question(retriever, question)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)