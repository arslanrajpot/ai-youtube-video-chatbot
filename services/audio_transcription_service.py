import yt_dlp
import tempfile
import os
import logging
import importlib
import shutil
from pathlib import Path
import subprocess
import threading
import wave

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioTranscriptionService:
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize the audio transcription service with Whisper.
        
        Args:
            model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                       - "base" is a good balance of speed and accuracy
        """
        self.model_size = model_size
        self.model = None
        self._transcribe_lock = threading.Lock()
        logger.info(f"Audio transcription service initialized (model: {model_size})")

    def _ensure_model_loaded(self):
        """Lazy-load Whisper so app startup does not fail on restricted systems."""
        if self.model is not None:
            return
        try:
            whisper = importlib.import_module("whisper")
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Whisper initialization failed: {str(e)}")
            raise Exception(
                "Whisper could not be initialized on this machine. "
                "You can still use videos with built-in YouTube transcripts. "
                f"Underlying error: {str(e)}"
            )

    def _resolve_ffmpeg(self) -> str | None:
        """
        Resolve a usable ffmpeg executable path.
        Tries PATH first, then common local locations (including winget install dir).
        """
        found = shutil.which("ffmpeg")
        if found:
            return found

        candidates: list[Path] = []
        cwd = Path.cwd()
        candidates.append(cwd / "ffmpeg.exe")
        candidates.append(cwd / "ffmpeg")

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_ffmpeg_dir = (
                Path(local_app_data)
                / "Microsoft"
                / "WinGet"
                / "Packages"
                / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            )
            if winget_ffmpeg_dir.exists():
                for p in winget_ffmpeg_dir.rglob("ffmpeg.exe"):
                    candidates.append(p)

        for c in candidates:
            if c.exists() and c.is_file():
                ff_dir = str(c.parent)
                os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
                return str(c)
        return None
    
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
            self._ensure_model_loaded()
            ffmpeg_bin = self._resolve_ffmpeg()
            if not ffmpeg_bin:
                raise Exception(
                    "FFmpeg executable was not found. Install FFmpeg and ensure `ffmpeg -version` works "
                    "or place ffmpeg.exe in the project root."
                )
            logger.info(f"Using FFmpeg binary: {ffmpeg_bin}")

            # Extract audio from YouTube
            audio_path = self._extract_audio(video_id, max_duration)
            logger.info(f"Audio extracted to: {audio_path}")

            # Normalize to a deterministic PCM WAV before Whisper.
            # Some downloaded containers/codecs can produce empty mel segments in Whisper.
            audio_path = self._normalize_audio_for_whisper(audio_path)
            logger.info(f"Audio normalized for Whisper: {audio_path}")
            
            # Transcribe with Whisper
            logger.info("Starting transcription with Whisper...")
            try:
                # Whisper model decode is not reliably re-entrant on CPU in this setup.
                # Serialize transcriptions to avoid intermittent tensor-shape runtime errors.
                with self._transcribe_lock:
                    result = self.model.transcribe(audio_path)
                transcript_text = result["text"].strip()
            except Exception as e:
                if isinstance(e, FileNotFoundError) or "WinError 2" in str(e):
                    raise Exception(
                        "FFmpeg executable could not be launched. Install FFmpeg and ensure "
                        "`ffmpeg -version` works in the same terminal."
                    )
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
        
        base_path = temp_path.replace(".wav", "")

        def _cleanup_partial():
            for ext in [".m4a", ".mp4", ".webm", ".wav", ".opus", ".ogg"]:
                p = base_path + ext
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        # YouTube 403: rotate clients; keep yt-dlp updated (pip install -U yt-dlp)
        player_tries = (
            ["android"],
            ["web"],
            ["ios"],
            ["mweb"],
            ["android", "web"],
        )
        cookiefile = os.environ.get("YTDLP_COOKIEFILE") or os.environ.get(
            "YOUTUBE_COOKIEFILE"
        )
        last_error: Exception | None = None

        for clients in player_tries:
            _cleanup_partial()
            ydl_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": temp_path.replace(".wav", ".%(ext)s"),
                "noplaylist": True,
                "max_duration": max_duration,
                "quiet": True,
                "no_warnings": True,
                "extractor_args": {"youtube": {"player_client": clients}},
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            }
            if cookiefile and os.path.isfile(cookiefile):
                ydl_opts["cookiefile"] = cookiefile

            try:
                logger.info(
                    f"Extracting audio from: {url} (yt-dlp player_client={clients})"
                )
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                downloaded_file = None
                for ext in [".m4a", ".mp4", ".webm", ".wav", ".opus", ".ogg"]:
                    potential_file = base_path + ext
                    if os.path.exists(potential_file):
                        downloaded_file = potential_file
                        break
                if not downloaded_file or os.path.getsize(downloaded_file) == 0:
                    raise RuntimeError("No audio file after download")
                logger.info(
                    f"Audio extraction successful. File: {downloaded_file}, "
                    f"Size: {os.path.getsize(downloaded_file)} bytes"
                )
                return downloaded_file
            except Exception as e:
                last_error = e
                logger.warning(
                    f"yt-dlp failed with client {clients}: {e!r}"
                )
                continue

        _cleanup_partial()
        err_msg = str(last_error) if last_error else "unknown error"
        if "403" in err_msg and not (
            os.environ.get("YTDLP_COOKIEFILE")
            or os.environ.get("YOUTUBE_COOKIEFILE")
        ):
            err_msg += (
                " | Tip: export browser cookies to cookies.txt, set YTDLP_COOKIEFILE to its path, "
                "and run: pip install -U yt-dlp"
            )
        logger.error(f"Audio extraction failed after all clients: {err_msg}")
        raise Exception(f"Failed to extract audio from YouTube: {err_msg}")

    def _normalize_audio_for_whisper(self, input_path: str) -> str:
        """
        Convert arbitrary media/audio input to 16kHz mono PCM WAV for Whisper stability.
        """
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise Exception("Downloaded audio file is missing or empty")

        out_fd, out_path = tempfile.mkstemp(suffix="_whisper.wav")
        os.close(out_fd)
        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            out_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0:
                stderr = (proc.stderr or "").strip()
                raise Exception(
                    "FFmpeg could not convert downloaded media to WAV for Whisper. "
                    f"Return code={proc.returncode}. stderr={stderr[:500]}"
                )
            # Validate resulting WAV has usable audio frames.
            with wave.open(out_path, "rb") as wf:
                nframes = wf.getnframes()
                fr = wf.getframerate()
                duration = (nframes / float(fr)) if fr else 0.0
            if nframes <= 0 or duration < 0.25:
                raise Exception(
                    f"Normalized audio is empty/too short for Whisper (frames={nframes}, duration={duration:.3f}s)"
                )
        finally:
            # Keep input cleanup local to this stage to avoid leaking temporary downloads.
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except OSError:
                    pass

        return out_path
    
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
