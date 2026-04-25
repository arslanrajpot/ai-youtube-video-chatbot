from __future__ import annotations

import logging
import os
import traceback
import threading
from typing import Any

from flask import Flask, request, render_template, jsonify
import services.youtube_service as youtube_service_module
from services.youtube_service import YOUTUBE_SERVICE_REV, YouTubeService
from services.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Included in every JSON response so the UI/Network tab proves this code path ran.
API_BUILD = "v2-2025-04-25"

# Set YOUTUBE_CHATBOT_DEBUG=0 in production to hide tracebacks in JSON (types/causes are still returned).
YOUTUBE_CHATBOT_DEBUG = os.environ.get("YOUTUBE_CHATBOT_DEBUG", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _chained_causes(exc: BaseException) -> list[dict[str, str]]:
    """Walk explicit __cause__ then, if still needed, the original __context__ once."""
    out: list[dict[str, str]] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(
            {
                "type": type(cur).__name__,
                "module": type(cur).__module__,
                "str": str(cur),
            }
        )
        nxt: BaseException | None = cur.__cause__
        if nxt is None and cur.__context__ is not None and id(cur.__context__) not in seen:
            nxt = cur.__context__
        cur = nxt
    return out


def _debug_error(exc: BaseException, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "error_message": str(exc),
        "causes": _chained_causes(exc),
        "youtube_service_py": getattr(youtube_service_module, "__file__", None),
    }
    data.update({k: v for k, v in extra.items() if v is not None})
    if YOUTUBE_CHATBOT_DEBUG:
        data["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return data


@app.after_request
def _no_cache_api(res):
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    res.headers["Pragma"] = "no-cache"
    return res

# Confirms which source file is loaded (helps when multiple copies of the repo exist)
print(f"[startup] youtube_service module: {youtube_service_module.__file__}", flush=True)

youtube_service = YouTubeService()
# Don't pass Google API key to avoid quota issues - let RAG service handle fallback
rag_service = RAGService(api_key=None)
_processing_videos: set[str] = set()
_processing_lock = threading.Lock()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", api_build=API_BUILD)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "YouTube Talker",
            "app_build": API_BUILD,
            "debug_json": YOUTUBE_CHATBOT_DEBUG,
            "pid": os.getpid(),
        }
    )


@app.route("/debug/server", methods=["GET"])
def debug_server():
    """Open in browser to verify you are hitting this Flask process and the expected code on disk."""
    return jsonify(
        {
            "youtube_service_py": getattr(youtube_service_module, "__file__", None),
            "transcript_fetch_marker": "v2_get_transcript_list_transcripts",
            "youtube_service_rev": YOUTUBE_SERVICE_REV,
            "app_build": API_BUILD,
            "debug_json": YOUTUBE_CHATBOT_DEBUG,
            "pid": os.getpid(),
        }
    )

@app.route("/submit_video", methods=["POST"])
def submit_video():
    # Visible proof in the terminal that this Flask process received the form (not another server/port).
    print(f"[submit_video] from {request.remote_addr} body keys={list(request.form.keys())}", flush=True)
    youtube_url = request.form.get("youtube_url", "")
    video_id = None
    try:
        if not youtube_url:
            raise ValueError("Missing youtube_url in form")
        video_id = youtube_service.extract_video_id(youtube_url)
        print(f"[submit_video] video_id={video_id}", flush=True)
        with _processing_lock:
            if video_id in _processing_videos:
                return jsonify(
                    {
                        "status": "error",
                        "message": "This video is already being processed. Please wait for the current job to finish.",
                        "app_build": API_BUILD,
                        "debug": {"video_id": video_id, "reason": "duplicate_inflight_submit"},
                    }
                ), 409
            _processing_videos.add(video_id)
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
            "llm_provider": getattr(rag_service, 'llm_provider', 'unknown'),
            "app_build": API_BUILD,
        })
    except Exception as e:
        logger.exception("submit_video failed")
        dbg = _debug_error(
            e,
            path="/submit_video",
            remote_addr=request.remote_addr,
            form_keys=list(request.form.keys()),
            youtube_url_len=len(youtube_url),
            video_id=video_id,
        )
        return jsonify(
            {
                "status": "error",
                "message": str(e),
                "app_build": API_BUILD,
                "debug": dbg,
            }
        ), 400
    finally:
        if video_id:
            with _processing_lock:
                _processing_videos.discard(video_id)

@app.route("/ask_question", methods=["POST"])
def ask_question():
    question = request.form.get("question", "")
    video_id = request.form.get("video_id", "")
    try:
        if not rag_service.groq_api_key and not rag_service.google_api_key:
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "LLM is not configured. Set GROQ_API_KEY (recommended) "
                        "or GOOGLE_API_KEY in .env, then restart the server."
                    ),
                    "app_build": API_BUILD,
                    "debug": {
                        "path": "/ask_question",
                        "missing_groq_api_key": not bool(rag_service.groq_api_key),
                        "missing_google_api_key": not bool(rag_service.google_api_key),
                    },
                }
            ), 400

        retriever = rag_service.get_retriever(video_id)
        if not retriever:
            return jsonify(
                {
                    "status": "error",
                    "message": "No video transcript stored. Please submit a video first.",
                    "app_build": API_BUILD,
                    "debug": {
                        "path": "/ask_question",
                        "video_id_empty": not bool(video_id),
                        "question_len": len(question),
                    },
                }
            ), 400
        answer = rag_service.answer_question(retriever, question)
        return jsonify(
            {
                "status": "success",
                "answer": answer,
                "app_build": API_BUILD,
            }
        )
    except Exception as e:
        logger.exception("ask_question failed")
        dbg = _debug_error(
            e,
            path="/ask_question",
            remote_addr=request.remote_addr,
            form_keys=list(request.form.keys()),
            video_id=video_id or None,
            question_len=len(question),
        )
        return jsonify(
            {
                "status": "error",
                "message": str(e),
                "app_build": API_BUILD,
                "debug": dbg,
            }
        ), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)