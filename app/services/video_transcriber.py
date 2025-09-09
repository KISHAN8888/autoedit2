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
from groq import Groq, RateLimitError
from pydub import AudioSegment
import subprocess
import re
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VideoTranscriber:
    """Async video transcriber that handles multiple concurrent requests"""
    
    def __init__(self, transcription_method="local", model_size="small", api_key=None, groq_api_key=None, language="en"):
        self.transcription_method = transcription_method
        self.model_size = model_size
        self.api_key = api_key
        self.language = language
        self.groq_api_key = groq_api_key


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

    async def transcribe_with_groq(self, audio_path, chunk_length=600, overlap=10):
        """
        Async Groq transcription with chunking support
        
        Args:
            audio_path: Path to audio file
            chunk_length: Length of each chunk in seconds (default: 600)
            overlap: Overlap between chunks in seconds (default: 10)
            
        Returns:
            Tuple of (language, srt_content)
        """
        try:
            if not self.groq_api_key:
                raise ValueError("Groq API key not set")
                
            logger.info(f"Starting Groq transcription with chunking: {audio_path}")
            
            # Initialize the Groq client
            client = Groq(api_key=self.groq_api_key, max_retries=0)
            
            # Preprocess audio to optimal format
            processed_path = await self._preprocess_audio_for_groq(audio_path)
            
            try:
                # Load audio with pydub
                audio = AudioSegment.from_file(processed_path, format="flac")
                duration = len(audio)  # Duration in milliseconds
                
                logger.info(f"Audio duration: {duration/1000:.2f}s")
                
                # Calculate chunk parameters
                chunk_ms = chunk_length * 1000
                overlap_ms = overlap * 1000
                total_chunks = max(1, (duration // (chunk_ms - overlap_ms)) + 1)
                
                logger.info(f"Processing {total_chunks} chunks...")
                
                # Process chunks
                results = []
                total_transcription_time = 0
                
                for i in range(total_chunks):
                    start = i * (chunk_ms - overlap_ms)
                    end = min(start + chunk_ms, duration)
                    
                    logger.info(f"Processing chunk {i+1}/{total_chunks} - Time range: {start/1000:.1f}s - {end/1000:.1f}s")
                    
                    chunk = audio[start:end]
                    result, chunk_time = await self._transcribe_single_chunk_groq(client, chunk, i+1, total_chunks)
                    total_transcription_time += chunk_time
                    results.append((result, start))
                
                # Merge transcription results
                final_result = self._merge_groq_transcripts(results)
                
                # Extract language and convert to SRT format
                language = final_result.get('language', 'en')
                
                # Convert to SRT format using segments if available, otherwise use text
                if 'segments' in final_result and final_result['segments']:
                    srt_content = self.segments_to_srt_content(final_result['segments'])
                else:
                    # Fallback to simple text formatting
                    srt_content = self.normalize_srt_format(final_result['text'])
                
                logger.info(f"Groq transcription successful. Language: {language}, Content length: {len(srt_content)} chars")
                logger.info(f"Total Groq API transcription time: {total_transcription_time:.2f}s")
                
                return language, srt_content
                
            finally:
                # Clean up temporary preprocessed file
                if processed_path and Path(processed_path).exists():
                    Path(processed_path).unlink(missing_ok=True)
                    
        except Exception as e:
            logger.error(f"Error in Groq transcription: {e}", exc_info=True)
            raise

    async def _preprocess_audio_for_groq(self, input_path):
        """
        Preprocess audio file to 16kHz mono FLAC using ffmpeg for optimal Groq processing.
        """
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            output_path = temp_file.name
        
        logger.info("Converting audio to 16kHz mono FLAC for Groq...")
        
        try:
            # Use asyncio subprocess for non-blocking operation
            process = await asyncio.create_subprocess_exec(
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'flac',
                '-y',
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                Path(output_path).unlink(missing_ok=True)
                raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()}")
                
            return output_path
            
        except Exception as e:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError(f"Audio preprocessing failed: {e}")

    async def _transcribe_single_chunk_groq(self, client, chunk, chunk_num, total_chunks):
        """
        Transcribe a single audio chunk with Groq API with rate limiting handling.
        
        Returns:
            Tuple of (transcription result, processing time)
        """
        total_api_time = 0
        
        while True:
            # Create a temporary file that we can properly manage
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Export chunk to temporary file
                chunk.export(temp_path, format='flac')
                
                start_time = time.time()
                try:
                    with open(temp_path, 'rb') as file:
                        result = client.audio.transcriptions.create(
                            file=file,
                            model="whisper-large-v3",
                            language="en",
                            response_format="verbose_json",
                            timestamp_granularities=["word", "segment"]
                        )
                    
                    api_time = time.time() - start_time
                    total_api_time += api_time
                    
                    logger.info(f"Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                    return result, total_api_time
                    
                except RateLimitError as e:
                    logger.warning(f"Rate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                    await asyncio.sleep(60)  # Use async sleep
                    continue
                    
                except Exception as e:
                    logger.error(f"Error transcribing chunk {chunk_num}: {str(e)}")
                    raise
                    
            finally:
                # Clean up the temporary file
                if Path(temp_path).exists():
                    Path(temp_path).unlink(missing_ok=True)

    def _find_longest_common_sequence(self, sequences, match_by_words=True):
        """
        Find the optimal alignment between sequences with longest common sequence.
        """
        if not sequences:
            return ""

        # Convert input based on matching strategy
        if match_by_words:
            sequences = [
                [word for word in re.split(r'(\s+\w+)', seq) if word]
                for seq in sequences
            ]
        else:
            sequences = [list(seq) for seq in sequences]

        left_sequence = sequences[0]
        left_length = len(left_sequence)
        total_sequence = []

        for right_sequence in sequences[1:]:
            max_matching = 0.0
            right_length = len(right_sequence)
            max_indices = (left_length, left_length, 0, 0)

            # Try different alignments
            for i in range(1, left_length + right_length + 1):
                eps = float(i) / 10000.0

                left_start = max(0, left_length - i)
                left_stop = min(left_length, left_length + right_length - i)
                left = left_sequence[left_start:left_stop]

                right_start = max(0, i - left_length)
                right_stop = min(right_length, i)
                right = right_sequence[right_start:right_stop]

                if len(left) != len(right):
                    raise RuntimeError("Mismatched subsequences detected during transcript merging.")

                matches = sum(a == b for a, b in zip(left, right))
                matching = matches / float(i) + eps

                if matches > 1 and matching > max_matching:
                    max_matching = matching
                    max_indices = (left_start, left_stop, right_start, right_stop)

            # Use the best alignment found
            left_start, left_stop, right_start, right_stop = max_indices
            
            left_mid = (left_stop + left_start) // 2
            right_mid = (right_stop + right_start) // 2
            
            total_sequence.extend(left_sequence[:left_mid])
            left_sequence = right_sequence[right_mid:]
            left_length = len(left_sequence)

        # Add remaining sequence
        total_sequence.extend(left_sequence)
        
        # Join back into text
        if match_by_words:
            return ''.join(total_sequence)
        return ''.join(total_sequence)

    def _merge_groq_transcripts(self, results):
        """
        Merge transcription chunks and handle overlaps for Groq API responses.
        
        Args:
            results: List of (result, start_time) tuples
            
        Returns:
            dict: Merged transcription with text, segments, and words
        """
        logger.info("Merging Groq transcription results...")
        
        # Check if we have segments in our results
        has_segments = False
        for chunk, _ in results:
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
            if 'segments' in data and data['segments'] is not None and len(data['segments']) > 0:
                has_segments = True
                break
        
        # Process word-level timestamps
        has_words = False
        words = []
        
        for chunk, chunk_start_ms in results:
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
            
            if isinstance(data, dict) and 'words' in data and data['words'] is not None and len(data['words']) > 0:
                has_words = True
                chunk_words = data['words']
                for word in chunk_words:
                    word['start'] = word['start'] + (chunk_start_ms / 1000)
                    word['end'] = word['end'] + (chunk_start_ms / 1000)
                words.extend(chunk_words)
        
        # If no segments, merge text only
        if not has_segments:
            logger.info("No segments found in transcription results. Merging full texts only.")
            
            texts = []
            for chunk, _ in results:
                data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
                text = data.get('text', '') if isinstance(data, dict) else getattr(chunk, 'text', '')
                texts.append(text)
            
            merged_text = " ".join(texts)
            result = {"text": merged_text, "segments": []}
            
            if has_words:
                result["words"] = words
                
            return result
        
        # Merge segments with overlap handling
        logger.info("Merging segments across chunks...")
        final_segments = []
        processed_chunks = []
        
        for i, (chunk, chunk_start_ms) in enumerate(results):
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
            segments = data.get('segments', []) if isinstance(data, dict) else getattr(chunk, 'segments', [])
            
            # Convert segments to list of dicts if needed
            if hasattr(segments, 'model_dump'):
                segments = segments.model_dump()
            elif not isinstance(segments, list):
                segments = []
            
            # Handle overlap with next chunk
            if i < len(results) - 1:
                next_start = results[i + 1][1]  # Next chunk start time in ms
                
                current_segments = []
                overlap_segments = []
                
                for segment in segments:
                    if not isinstance(segment, dict) and hasattr(segment, 'model_dump'):
                        segment = segment.model_dump()
                    elif not isinstance(segment, dict):
                        segment = {
                            'text': getattr(segment, 'text', ''),
                            'start': getattr(segment, 'start', 0),
                            'end': getattr(segment, 'end', 0)
                        }
                    
                    segment_end = segment['end']
                    
                    if segment_end * 1000 > next_start:
                        overlap_segments.append(segment)
                    else:
                        current_segments.append(segment)
                
                # Merge overlap segments if any exist
                if overlap_segments:
                    merged_overlap = overlap_segments[0].copy()
                    merged_overlap.update({
                        'text': ' '.join(s.get('text', '') for s in overlap_segments),
                        'end': overlap_segments[-1].get('end', 0)
                    })
                    current_segments.append(merged_overlap)
                    
                processed_chunks.append(current_segments)
            else:
                # Last chunk - ensure all segments are dicts
                dict_segments = []
                for segment in segments:
                    if not isinstance(segment, dict) and hasattr(segment, 'model_dump'):
                        dict_segments.append(segment.model_dump())
                    elif not isinstance(segment, dict):
                        dict_segments.append({
                            'text': getattr(segment, 'text', ''),
                            'start': getattr(segment, 'start', 0),
                            'end': getattr(segment, 'end', 0)
                        })
                    else:
                        dict_segments.append(segment)
                processed_chunks.append(dict_segments)
        
        # Merge boundaries between chunks
        for i in range(len(processed_chunks) - 1):
            if not processed_chunks[i] or not processed_chunks[i+1]:
                continue
                
            if len(processed_chunks[i]) > 1:
                final_segments.extend(processed_chunks[i][:-1])
            
            # Merge boundary segments
            last_segment = processed_chunks[i][-1]
            first_segment = processed_chunks[i+1][0]
            
            merged_text = self._find_longest_common_sequence([
                last_segment.get('text', ''),
                first_segment.get('text', '')
            ])
            
            merged_segment = last_segment.copy()
            merged_segment.update({
                'text': merged_text,
                'end': first_segment.get('end', 0)
            })
            final_segments.append(merged_segment)
        
        # Add all segments from last chunk
        if processed_chunks and processed_chunks[-1]:
            final_segments.extend(processed_chunks[-1])
        
        # Create final transcription
        final_text = ' '.join(segment.get('text', '') for segment in final_segments)
        
        result = {
            "text": final_text,
            "segments": final_segments
        }
        
        if has_words:
            result["words"] = words
        
        return result
    
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
            # Handle both dict and object formats
            if isinstance(segment, dict):
                start_time = self.format_time(segment['start'])
                end_time = self.format_time(segment['end'])
                text = segment['text']
            else:
                # Handle object format (legacy support)
                start_time = self.format_time(segment.start)
                end_time = self.format_time(segment.end)
                text = segment.text
            
            srt_content += f"{index}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text.strip()}\n\n"
        
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
                elif self.transcription_method == "local":
                    language, segments = await self.transcribe_audio_local_async(audio_path)
                    # Convert segments to SRT content
                    srt_content = self.segments_to_srt_content(segments)
                elif self.transcription_method == "groq":
                    language, srt_content = await self.transcribe_with_groq(audio_path)
                    # Normalize Groq response
                    srt_content = self.normalize_srt_format(srt_content)
                logger.info(f"Transcription successful. Language: {language}, Content length: {len(srt_content)} chars")
                return language, srt_content
                
            except Exception as e:
                logger.error(f"Error in transcription: {e}")
                raise
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# # EXAMPLE USAGE:
# async def main():
#     video_path = r"C:\Users\amand\Videos\2025-03-19 01-56-12.mkv"
    
#     # Check if file exists first
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found at {video_path}")
#         return
    
#     print(f"Video file found: {video_path}")
#     print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
#     # Provide your API key
#     api_key = "groq_api_key"
#     transcriber = VideoTranscriber(transcription_method="groq", groq_api_key=api_key)
    
#     try:
#         # Call the method on the instance
#         language, srt_content = await transcriber.transcribe_video_async(video_path) 
#         print(f"Language: {language}")
#         print(f"SRT content length: {len(srt_content)} characters")
        
#         if not srt_content.strip():
#             print("Warning: SRT content is empty or only whitespace")
#         else:
#             # Save SRT content to a file
#             srt_file_path = os.path.splitext(video_path)[0] + ".srt"
#             with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
#                 srt_file.write(srt_content)
#             print(f"SRT content saved to: {srt_file_path}")
            
#     except Exception as e:
#         print(f"Error during transcription: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
