from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from .audio_transcription_service import AudioTranscriptionService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

    def _fetch_youtube_transcript(self, video_id):
        """Fetch transcript using YouTube Transcript API."""
        try:
            logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
            api = YouTubeTranscriptApi()
            transcripts = api.list(video_id)
            logger.info(f"Available transcripts: {[t.language_code for t in transcripts]}")
            transcript = None
            for t in transcripts:
                if t.language_code == "en":
                    transcript = t.fetch()
                    logger.info(f"Raw transcript (first 100 chars): {str(transcript)[:100]}")
                    break
            if not transcript:
                logger.error(f"No English transcript available for video ID: {video_id}")
                raise TranscriptsDisabled("No English transcript available")
            transcript_text = " ".join([entry.text for entry in transcript])
            if not transcript_text.strip():
                logger.error(f"Empty transcript for video ID: {video_id}")
                raise ValueError("Transcript is empty or contains no valid text")
            return transcript_text
        except Exception as e:
            logger.error(f"YouTube API error for video ID: {video_id}: {str(e)}")
            raise ValueError(f"Failed to fetch transcript: {str(e)}")