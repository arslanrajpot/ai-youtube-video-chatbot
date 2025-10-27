import whisper
import yt_dlp
import tempfile
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioTranscriptionService:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the audio transcription service with Whisper.
        
        Args:
            model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                       - "base" is a good balance of speed and accuracy
        """
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")
    
    def extract_audio_and_transcribe(self, video_id: str, max_duration: int = 3600) -> str:
        """
        Extract audio from YouTube video and transcribe it using Whisper.
        
        Args:
            video_id: YouTube video ID
            max_duration: Maximum duration in seconds (default: 1 hour)
            
        Returns:
            Transcribed text
            
        Raises:
            Exception: If audio extraction or transcription fails
        """
        try:
            # Extract audio from YouTube
            audio_path = self._extract_audio(video_id, max_duration)
            logger.info(f"Audio extracted to: {audio_path}")
            
            # Transcribe with Whisper
            logger.info("Starting transcription with Whisper...")
            try:
                result = self.model.transcribe(audio_path)
                transcript_text = result["text"].strip()
            except Exception as e:
                if "ffmpeg" in str(e).lower():
                    logger.error("FFmpeg not available. Audio transcription requires FFmpeg.")
                    raise Exception("Audio transcription requires FFmpeg. Please install FFmpeg using: brew install ffmpeg (or use videos with YouTube transcripts)")
                else:
                    raise e
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary audio file cleaned up")
            
            if not transcript_text:
                raise ValueError("Transcription resulted in empty text")
            
            logger.info(f"Transcription completed. Length: {len(transcript_text)} characters")
            return transcript_text
            
        except Exception as e:
            # Clean up audio file if it exists
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            logger.error(f"Audio transcription failed for video {video_id}: {str(e)}")
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    def _extract_audio(self, video_id: str, max_duration: int) -> str:
        """
        Extract audio from YouTube video using yt-dlp.
        
        Args:
            video_id: YouTube video ID
            max_duration: Maximum duration in seconds
            
        Returns:
            Path to temporary audio file
        """
        # Create temporary file for audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()
        
        # YouTube URL
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # yt-dlp options - simplified without FFmpeg
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': temp_path.replace('.wav', '.%(ext)s'),
            'noplaylist': True,
            'max_duration': max_duration,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            logger.info(f"Extracting audio from: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the actual downloaded file (it might have a different extension)
            downloaded_file = None
            base_path = temp_path.replace('.wav', '')
            for ext in ['.m4a', '.mp4', '.webm', '.wav']:
                potential_file = base_path + ext
                if os.path.exists(potential_file):
                    downloaded_file = potential_file
                    break
            
            if not downloaded_file or os.path.getsize(downloaded_file) == 0:
                raise Exception("Audio extraction failed - no audio file created")
            
            logger.info(f"Audio extraction successful. File: {downloaded_file}, Size: {os.path.getsize(downloaded_file)} bytes")
            return downloaded_file
            
        except Exception as e:
            # Clean up temp files if they exist
            base_path = temp_path.replace('.wav', '')
            for ext in ['.m4a', '.mp4', '.webm', '.wav']:
                potential_file = base_path + ext
                if os.path.exists(potential_file):
                    os.remove(potential_file)
            logger.error(f"Audio extraction failed: {str(e)}")
            raise Exception(f"Failed to extract audio from YouTube: {str(e)}")
    
    def get_available_models(self) -> list:
        """Get list of available Whisper models."""
        return ["tiny", "base", "small", "medium", "large"]
    
    def estimate_transcription_time(self, duration_seconds: int) -> int:
        """
        Estimate transcription time based on video duration.
        Whisper typically processes audio at ~10x real-time speed.
        
        Args:
            duration_seconds: Video duration in seconds
            
        Returns:
            Estimated transcription time in seconds
        """
        return max(10, duration_seconds // 10)  # Minimum 10 seconds
