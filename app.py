from flask import Flask, request, render_template, jsonify
from services.youtube_service import YouTubeService
from services.rag_service import RAGService
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
youtube_service = YouTubeService()
# Don't pass Google API key to avoid quota issues - let RAG service handle fallback
rag_service = RAGService(api_key=None)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "YouTube Talker"})

@app.route("/submit_video", methods=["POST"])
def submit_video():
    youtube_url = request.form["youtube_url"]
    try:
        video_id = youtube_service.extract_video_id(youtube_url)
        transcript, source_type = youtube_service.fetch_transcript(video_id)
        rag_service.process_transcript(transcript, video_id)
        
        # Provide informative message based on source
        if source_type == "youtube_transcript":
            message = "Video transcript stored successfully (from YouTube captions)"
        else:
            message = "Video transcript stored successfully (from AI audio transcription)"
        
        return jsonify({
            "status": "success", 
            "message": message,
            "source": source_type,
            "llm_provider": getattr(rag_service, 'llm_provider', 'unknown')
        })
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
    # For PythonAnywhere, use default settings
    app.run(debug=False)