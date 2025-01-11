from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class YouTubeService:
    @staticmethod
    def extract_video_id(url):
        if "v=" not in url:
            raise ValueError("Invalid YouTube URL: Missing 'v=' parameter")
        return url.split("v=")[1].split("&")[0]

    @staticmethod
    def fetch_transcript(video_id):
        try:
            logger.info(f"Fetching transcript for video ID: {video_id}")
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