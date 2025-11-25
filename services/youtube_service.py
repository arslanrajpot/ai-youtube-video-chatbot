import logging
import os
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from .audio_transcription_service import AudioTranscriptionService

load_dotenv()

# Bumped when transcript logic changes; helps confirm the running process is not using stale code.
YOUTUBE_SERVICE_REV = 3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _transcript_cookies_path() -> str | None:
    """Optional Netscape cookies.txt (same idea as yt-dlp). Unlocks some transcript / age-gated cases."""
    p = (os.environ.get("YOUTUBE_COOKIEFILE") or os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES") or "").strip()
    if p and os.path.isfile(p):
        return p
    if p and not os.path.isfile(p):
        logger.warning("YOUTUBE_COOKIEFILE is set but file not found: %s", p)
    return None

class YouTubeService:
    def __init__(self):
        """Initialize YouTube service with audio transcription fallback."""
        self.audio_transcription_service = AudioTranscriptionService()
    
    @staticmethod
    def extract_video_id(url):
        if "v=" not in url:
            raise ValueError("Invalid YouTube URL: Missing 'v=' parameter")
        return url.split("v=")[1].split("&")[0]

    def fetch_transcript(self, video_id, use_audio_fallback=True):
        """
        Fetch transcript from YouTube, with optional audio transcription fallback.
        
        Args:
            video_id: YouTube video ID
            use_audio_fallback: Whether to use audio transcription if YouTube transcript fails
            
        Returns:
            tuple: (transcript_text, source_type)
            - transcript_text: The transcript content
            - source_type: "youtube_transcript" or "audio_transcription"
        """
        # Try YouTube transcript first
        try:
            transcript_text = self._fetch_youtube_transcript(video_id)
            logger.info(f"Successfully fetched YouTube transcript for video {video_id}")
            return transcript_text, "youtube_transcript"
        except Exception as e:
            logger.warning(f"YouTube transcript failed for video {video_id}: {str(e)}")
            
            if not use_audio_fallback:
                raise e
            
            # Fallback to audio transcription
            try:
                logger.info(f"Attempting audio transcription for video {video_id}")
                transcript_text = self.audio_transcription_service.extract_audio_and_transcribe(video_id)
                logger.info(f"Successfully transcribed audio for video {video_id}")
                return transcript_text, "audio_transcription"
            except Exception as audio_error:
                logger.error(f"Audio transcription also failed for video {video_id}: {str(audio_error)}")
                raise Exception(f"Both YouTube transcript and audio transcription failed. YouTube error: {str(e)}. Audio error: {str(audio_error)}")

    @staticmethod
    def _fetched_to_text(fetched) -> str:
        if not fetched:
            return ""
        first = fetched[0]
        if isinstance(first, dict):
            return " ".join(e["text"] for e in fetched)
        return " ".join(e.text for e in fetched)

    def _fetch_youtube_transcript(self, video_id):
        """Fetch transcript using YouTube Transcript API (no instance .list — use get_transcript / list_transcripts)."""
        try:
            cookies = _transcript_cookies_path()
            logger.info(
                "YouTube transcript rev=%s file=%s cookies=%s video_id=%s",
                YOUTUBE_SERVICE_REV,
                __file__,
                "yes" if cookies else "no",
                video_id,
            )
            # 1) Fast path: direct fetch (works with youtube-transcript-api 0.6.x+)
            try:
                data = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=["en", "en-US", "en-GB"],
                    cookies=cookies,
                )
                text = self._fetched_to_text(data)
                if text.strip():
                    logger.info(
                        f"get_transcript ok (first 100 chars): {str(text)[:100]}"
                    )
                    return text
            except Exception as ex:
                logger.info(f"get_transcript failed, trying list_transcripts: {ex}")

            # 2) list_transcripts + pick language
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies)
            logger.info(
                f"Available transcripts: {[(t.language_code, t.is_generated) for t in transcript_list]}"
            )
            try:
                transcript = transcript_list.find_transcript(
                    ["en", "en-US", "en-GB"]
                )
            except NoTranscriptFound:
                try:
                    transcript = next(iter(transcript_list))
                except StopIteration as si:
                    raise TranscriptsDisabled("No transcripts available") from si
            fetched = transcript.fetch()
            text = self._fetched_to_text(fetched)
            if not text.strip():
                raise ValueError("Transcript is empty or contains no valid text")
            logger.info(f"list_transcripts ok (first 100 chars): {str(text)[:100]}")
            return text
        except Exception as e:
            logger.error(f"YouTube API error for video ID: {video_id}: {str(e)}")
            if isinstance(e, ET.ParseError):
                raise ValueError(
                    "Failed to fetch transcript: YouTube returned malformed transcript data "
                    "(often geo/cookie/rate-limit related). Try setting YOUTUBE_COOKIEFILE "
                    "to a browser-exported cookies.txt."
                )
            raise ValueError(f"Failed to fetch transcript: {str(e)}")