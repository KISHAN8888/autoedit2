# import os
# import sys
# import math
# import asyncio
# import tempfile
# import shutil
# import ffmpeg
# from faster_whisper import WhisperModel
# from pathlib import Path
# import logging
# from concurrent.futures import ThreadPoolExecutor

# logger = logging.getLogger(__name__)

# class VideoTranscriber:
#     """Async video transcriber that handles multiple concurrent requests"""
    
#     def __init__(self, transcription_method="local", model_size="small", api_key=None, language="en"):
#         self.transcription_method = transcription_method
#         self.model_size = model_size
#         self.api_key = api_key
#         self.language = language
        
#         if transcription_method == "api" and not api_key:
#             raise ValueError("API key required for API transcription method")
        
#         # Thread pool for CPU-intensive tasks
#         self.executor = ThreadPoolExecutor(max_workers=4)
    
#     async def check_ffmpeg(self):
#         """Async check if FFmpeg is available"""
#         try:
#             loop = asyncio.get_event_loop()
#             process = await asyncio.create_subprocess_exec(
#                 'ffmpeg', '-version',
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )
#             await process.wait()
#             return process.returncode == 0
#         except FileNotFoundError:
#             return False
#         except Exception:
#             return False
    
#     async def extract_audio_async(self, input_video_path, output_format="wav", temp_dir=None):
#         """Extract audio from video asynchronously"""
#         if not await self.check_ffmpeg():
#             raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        
#         audio_filename = f"audio.{output_format}"
#         extracted_audio_path = os.path.join(temp_dir, audio_filename)
        
#         logger.info(f"Extracting audio from {input_video_path} as {output_format.upper()}")
        
#         try:
#             input_path = os.path.abspath(input_video_path)
            
#             # Build ffmpeg command
#             stream = ffmpeg.input(input_path)
#             if output_format == "mp3":
#                 stream = ffmpeg.output(stream, extracted_audio_path, acodec='mp3', audio_bitrate='192k')
#             else:  # wav
#                 stream = ffmpeg.output(stream, extracted_audio_path)
            
#             # Run ffmpeg asynchronously
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 self.executor,
#                 lambda: ffmpeg.run(stream, overwrite_output=True, quiet=True)
#             )
            
#             logger.info(f"Audio extracted to: {extracted_audio_path}")
#             return extracted_audio_path
            
#         except ffmpeg.Error as e:
#             logger.error(f"FFmpeg error during audio extraction: {e}")
#             if e.stderr:
#                 logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
#             raise
#         except Exception as e:
#             logger.error(f"Error extracting audio: {e}")
#             raise
    
#     async def transcribe_with_api_async(self, audio_path, use_speaker_labels=False):
#         """Async API transcription"""
#         try:
#             from openai import OpenAI
#         except ImportError:
#             raise ImportError("openai package not installed. Install with: pip install --upgrade openai")
        
#         logger.info("Transcribing with LemonFox API")
        
#         audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
#         logger.info(f"Audio file size: {audio_size:.1f} MB")
        
#         try:
#             client = OpenAI(
#                 api_key=self.api_key,
#                 base_url="https://api.lemonfox.ai/v1",
#             )
            
#             # Run API call in thread pool to avoid blocking
#             loop = asyncio.get_event_loop()
            
#             def make_api_call():
#                 with open(audio_path, "rb") as audio_file:
#                     return client.audio.transcriptions.create(
#                         model="whisper-1",
#                         file=audio_file,
#                         language=self.language,
#                         response_format="srt"
#                     )
            
#             transcript = await loop.run_in_executor(self.executor, make_api_call)
            
#             logger.info("API transcription complete")
#             srt_content = str(transcript) if transcript else ""
            
#             if srt_content:
#                 logger.info(f"Received SRT content: {len(srt_content)} characters")
#             else:
#                 logger.warning("API returned empty response")
            
#             return self.language, srt_content
            
#         except Exception as e:
#             logger.error(f"Error during API transcription: {e}")
#             if "401" in str(e) or "unauthorized" in str(e).lower():
#                 raise RuntimeError("API key authentication failed")
#             elif "413" in str(e) or "too large" in str(e).lower():
#                 raise RuntimeError("Audio file too large for API")
#             elif "timeout" in str(e).lower():
#                 raise RuntimeError("API request timed out")
#             raise
    
#     async def transcribe_audio_local_async(self, audio_path):
#         """Async local Whisper transcription"""
#         logger.info(f"Loading Whisper model ({self.model_size})")
        
#         try:
#             # Run heavy computation in thread pool
#             loop = asyncio.get_event_loop()
            
#             def transcribe_sync():
#                 model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
#                 segments, info = model.transcribe(
#                     audio_path,
#                     beam_size=5,
#                     word_timestamps=True,
#                     vad_filter=True,
#                     vad_parameters=dict(min_silence_duration_ms=500)
#                 )
#                 return info.language, list(segments)
            
#             language, segments_list = await loop.run_in_executor(self.executor, transcribe_sync)
            
#             logger.info(f"Detected language: {language}")
#             logger.info(f"Transcription complete! Total segments: {len(segments_list)}")
            
#             return language, segments_list
            
#         except Exception as e:
#             logger.error(f"Error during local transcription: {e}")
#             raise
    
#     def format_time(self, seconds):
#         """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
#         hours = math.floor(seconds / 3600)
#         seconds %= 3600
#         minutes = math.floor(seconds / 60)
#         seconds %= 60
#         milliseconds = round((seconds - math.floor(seconds)) * 1000)
#         seconds = math.floor(seconds)
        
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
#     def segments_to_srt_content(self, segments):
#         """Convert transcription segments to SRT content string"""
#         srt_content = ""
        
#         for index, segment in enumerate(segments, 1):
#             start_time = self.format_time(segment.start)
#             end_time = self.format_time(segment.end)
            
#             srt_content += f"{index}\n"
#             srt_content += f"{start_time} --> {end_time}\n"
#             srt_content += f"{segment.text.strip()}\n\n"
        
#         return srt_content
    
#     def normalize_srt_format(self, srt_content):
#         """Normalize SRT format to ensure consistent formatting"""
#         lines = [line.strip() for line in srt_content.strip().split('\n') if line.strip()]
        
#         normalized_lines = []
#         i = 0
        
#         while i < len(lines):
#             if lines[i].isdigit():
#                 normalized_lines.append(lines[i])
#                 i += 1
                
#                 if i < len(lines) and '-->' in lines[i]:
#                     normalized_lines.append(lines[i])
#                     i += 1
                    
#                     subtitle_text_lines = []
#                     while i < len(lines) and not lines[i].isdigit():
#                         subtitle_text_lines.append(lines[i])
#                         i += 1
                    
#                     normalized_lines.extend(subtitle_text_lines)
                    
#                     if i < len(lines):
#                         normalized_lines.append("")
#                 else:
#                     i += 1
#             else:
#                 i += 1
        
#         return '\n'.join(normalized_lines)
    
#     async def transcribe_video_async(self, input_video_path):
#         """
#         Main async method to transcribe video to SRT content
#         Returns: (language, srt_content) tuple
#         """
#         logger.info(f"Starting transcription of {input_video_path}")
        
#         if not os.path.exists(input_video_path):
#             raise FileNotFoundError(f"Input video file not found: {input_video_path}")
        
#         # Use temporary directory for this transcription session
#         with tempfile.TemporaryDirectory() as temp_dir:
#             try:
#                 # Extract audio to temporary directory
#                 audio_format = "mp3" if self.transcription_method == "api" else "wav"
#                 audio_path = await self.extract_audio_async(input_video_path, audio_format, temp_dir)
                
#                 # Transcribe audio
#                 if self.transcription_method == "api":
#                     language, srt_content = await self.transcribe_with_api_async(audio_path)
#                     # Normalize API response
#                     srt_content = self.normalize_srt_format(srt_content)
#                 else:
#                     language, segments = await self.transcribe_audio_local_async(audio_path)
#                     # Convert segments to SRT content
#                     srt_content = self.segments_to_srt_content(segments)
                
#                 logger.info(f"Transcription successful. Language: {language}, Content length: {len(srt_content)} chars")
#                 return language, srt_content
                
#             except Exception as e:
#                 logger.error(f"Error in transcription: {e}")
#                 raise
    
#     def __del__(self):
#         """Cleanup thread pool executor"""
#         if hasattr(self, 'executor'):
#             self.executor.shutdown(wait=False)

import os
import sys
import math
import asyncio
import tempfile
import shutil
import ffmpeg
from faster_whisper import WhisperModel
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VideoTranscriber:
    """Async video transcriber that handles multiple concurrent requests"""
    
    def __init__(self, transcription_method="local", model_size="small", api_key=None, language="en"):
        self.transcription_method = transcription_method
        self.model_size = model_size
        self.api_key = api_key
        self.language = language


        self.model =None
        
        if transcription_method == "api" and not api_key:
            raise ValueError("API key required for API transcription method")
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def check_ffmpeg_sync(self):
        """Synchronous check if FFmpeg is available (fallback)"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("FFmpeg executable not found in PATH")
            return False
        except Exception as e:
            logger.error(f"Error checking FFmpeg: {e}")
            return False

    async def check_ffmpeg(self):
        """Async check if FFmpeg is available with fallback"""
        try:
            # Try async method first
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return True
        except FileNotFoundError:
            logger.warning("Async FFmpeg check failed, trying sync method")
        except Exception as e:
            logger.warning(f"Async FFmpeg check error: {e}, trying sync method")
        
        # Fallback to sync method
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.check_ffmpeg_sync)
    
    async def extract_audio_async(self, input_video_path, output_format="wav", temp_dir=None):
        """Extract audio from video asynchronously"""
        # Add debug logging
        logger.info("Checking FFmpeg availability...")
        ffmpeg_available = await self.check_ffmpeg()
        logger.info(f"FFmpeg check result: {ffmpeg_available}")
        
        if not ffmpeg_available:
            # Try to provide helpful error message
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')
            logger.error(f"FFmpeg not found. shutil.which('ffmpeg') returned: {ffmpeg_path}")
            
            # Check if it's in common locations
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg"
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    logger.info(f"Found FFmpeg at: {path}")
                    break
            else:
                logger.error("FFmpeg not found in common installation paths")
            
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        
        audio_filename = f"audio.{output_format}"
        extracted_audio_path = os.path.join(temp_dir, audio_filename)
        
        logger.info(f"Extracting audio from {input_video_path} as {output_format.upper()}")
        
        try:
            input_path = os.path.abspath(input_video_path)
            
            # Build ffmpeg command
            stream = ffmpeg.input(input_path)
            if output_format == "mp3":
                stream = ffmpeg.output(stream, extracted_audio_path, acodec='mp3', audio_bitrate='192k')
            else:  # wav
                stream = ffmpeg.output(stream, extracted_audio_path)
            
            # Run ffmpeg asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: ffmpeg.run(stream, overwrite_output=True, quiet=True)
            )
            
            logger.info(f"Audio extracted to: {extracted_audio_path}")
            return extracted_audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error during audio extraction: {e}")
            if e.stderr:
                logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    async def transcribe_with_api_async(self, audio_path, use_speaker_labels=False):
        """Async API transcription"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install --upgrade openai")
        
        logger.info("Transcribing with LemonFox API")
        
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        logger.info(f"Audio file size: {audio_size:.1f} MB")
        
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.lemonfox.ai/v1",
            )
            
            # Run API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def make_api_call():
                with open(audio_path, "rb") as audio_file:
                    return client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=self.language,
                        response_format="srt"
                    )
            
            transcript = await loop.run_in_executor(self.executor, make_api_call)
            print(f"this is transcript{transcript}")
            
            logger.info("API transcription complete")
            
            # Fix: Handle the response properly
            if hasattr(transcript, 'text'):
                # If it's an object with a text attribute
                srt_content = transcript.text
            else:
                # If it's a string, remove quotes and unescape newlines
                srt_content = str(transcript).strip('"').replace('\\n', '\n')
            
            if srt_content:
                logger.info(f"Received SRT content: {len(srt_content)} characters")
            else:
                logger.warning("API returned empty response")
            
            return self.language, srt_content
            
        except Exception as e:
            logger.error(f"Error during API transcription: {e}")
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise RuntimeError("API key authentication failed")
            elif "413" in str(e) or "too large" in str(e).lower():
                raise RuntimeError("Audio file too large for API")
            elif "timeout" in str(e).lower():
                raise RuntimeError("API request timed out")
            raise

    async def transcribe_audio_local_async(self, audio_path):
        """
        Async local Whisper transcription.
        This version loads the model only once for efficiency and stability.
        """
        loop = asyncio.get_event_loop()

        # Define the synchronous, CPU-bound tasks
        def load_model_sync():
            """This function loads the model."""
            logger.info(f"Loading Whisper model ({self.model_size}) for the first time...")
            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded successfully.")
            return model

        def transcribe_sync():
            """This function runs the transcription using the pre-loaded model."""
            logger.info(f"Starting transcription for {audio_path}...")
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            # The list conversion is important to realize the generator
            return info.language, list(segments)

        try:
            # Check if the model is loaded. If not, load it in the thread pool.
            if self.model is None:
                # The lock ensures that if two requests arrive at the same time,
                # only one will load the model.
                async with asyncio.Lock():
                    if self.model is None:
                        self.model = await loop.run_in_executor(self.executor, load_model_sync)
            
            # Now that the model is guaranteed to be loaded, run transcription.
            language, segments_list = await loop.run_in_executor(self.executor, transcribe_sync)
            
            logger.info(f"Detected language: {language}")
            logger.info(f"Transcription complete! Total segments: {len(segments_list)}")
            
            return language, segments_list
            
        except Exception as e:
            logger.error(f"Error during local transcription: {e}", exc_info=True)
            raise    
    
    def format_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = math.floor(seconds / 3600)
        seconds %= 3600
        minutes = math.floor(seconds / 60)
        seconds %= 60
        milliseconds = round((seconds - math.floor(seconds)) * 1000)
        seconds = math.floor(seconds)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def segments_to_srt_content(self, segments):
        """Convert transcription segments to SRT content string"""
        srt_content = ""
        
        for index, segment in enumerate(segments, 1):
            start_time = self.format_time(segment.start)
            end_time = self.format_time(segment.end)
            
            srt_content += f"{index}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{segment.text.strip()}\n\n"
        
        return srt_content
    
    def normalize_srt_format(self, srt_content):
       """Normalize SRT format to ensure consistent formatting"""
       print(f"Input SRT content: {repr(srt_content)}")
       print(f"Input length: {len(srt_content)}")
       
       lines = [line.strip() for line in srt_content.strip().split('\n') if line.strip()]
       print(f"Lines after split and strip: {len(lines)}")
       print(f"First few lines: {lines[:5]}")
       
       normalized_lines = []
       i = 0
       
       while i < len(lines):
           if lines[i].isdigit():
               normalized_lines.append(lines[i])
               i += 1
               
               if i < len(lines) and '-->' in lines[i]:
                   normalized_lines.append(lines[i])
                   i += 1
                   
                   subtitle_text_lines = []
                   while i < len(lines) and not lines[i].isdigit():
                       subtitle_text_lines.append(lines[i])
                       i += 1
                   
                   normalized_lines.extend(subtitle_text_lines)
                   
                   if i < len(lines):
                       normalized_lines.append("")
               else:
                   i += 1
           else:
               i += 1
       
       result = '\n'.join(normalized_lines)
       print(f"Output SRT content: {repr(result)}")
       print(f"Output length: {len(result)}")
       
       return result
    
    async def transcribe_video_async(self, input_video_path):
        """
        Main async method to transcribe video to SRT content
        Returns: (language, srt_content) tuple
        """
        logger.info(f"Starting transcription of {input_video_path}")
        
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video file not found: {input_video_path}")
        
        # Use temporary directory for this transcription session
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract audio to temporary directory
                audio_format = "mp3" if self.transcription_method == "api" else "wav"
                audio_path = await self.extract_audio_async(input_video_path, audio_format, temp_dir)
                
                # Transcribe audio
                if self.transcription_method == "api":
                    language, srt_content = await self.transcribe_with_api_async(audio_path)
                    # Normalize API response
                    srt_content = self.normalize_srt_format(srt_content)
                else:
                    language, segments = await self.transcribe_audio_local_async(audio_path)
                    # Convert segments to SRT content
                    srt_content = self.segments_to_srt_content(segments)
                
                logger.info(f"Transcription successful. Language: {language}, Content length: {len(srt_content)} chars")
                return language, srt_content
                
            except Exception as e:
                logger.error(f"Error in transcription: {e}")
                raise
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

async def main():
    video_path = r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4"
    
    # Check if file exists first
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print(f"Video file found: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Provide your API key
    api_key = "5cCrTnA7Nv73iujYaIxW302WLbfuVCnR"
    transcriber = VideoTranscriber(transcription_method="api", api_key=api_key)
    
    try:
        # Call the method on the instance
        language, srt_content = await transcriber.transcribe_video_async(video_path) 
        print(f"Language: {language}")
        print(f"SRT content length: {len(srt_content)} characters")
        print(f"SRT content: {repr(srt_content)}")  # repr() shows exact content including whitespace
        
        if not srt_content.strip():
            print("Warning: SRT content is empty or only whitespace")
        else:
            print(f"SRT: {srt_content}")
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())