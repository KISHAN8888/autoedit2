# import re
# import json
# import os
# import logging
# import time
# import asyncio
# import tempfile
# import shutil
# from dataclasses import dataclass
# from typing import List, Dict
# from pathlib import Path
# import openai
# import subprocess
# from pydantic import BaseModel
# import requests
# import aiohttp
# from requests.exceptions import RequestException
# from murf import Murf
# from gtts import gTTS
# import tempfile
# from PIL import Image, ImageDraw
# from concurrent.futures import ThreadPoolExecutor

# logger = logging.getLogger(__name__)

# # OpenAI Model Pricing (USD per 1 million tokens)
# MODEL_PRICING = {
#     "gpt-4o-mini": {"prompt_per_million": 0.15, "completion_per_million": 0.6},
# }

# # Pydantic models for OpenAI structured outputs
# class VideoAnalysis(BaseModel):
#     """Structured model for video analysis response"""
#     main_theme: str
#     key_sections: List[str]
#     important_terms: List[str]
#     tone: str
#     transition_markers: List[str]

# class SegmentOptimization(BaseModel):
#     """Structured model for segment optimization response"""
#     optimized_text: str
#     reasoning: str
#     confidence: float
#     preserved_terms: List[str]

# @dataclass
# class SRTSegment:
#     """Represents a single SRT subtitle segment"""
#     index: int
#     start_time: str
#     end_time: str
#     start_seconds: float
#     end_seconds: float
#     duration: float
#     text: str
#     word_count: int
#     words_per_second: float

# @dataclass
# class OptimizedSegment:
#     """Represents an optimized segment with new script and timing"""
#     original: SRTSegment
#     optimized_text: str
#     optimized_word_count: int
#     estimated_speech_duration: float
#     speed_multiplier: float
#     reasoning: str

# class VideoSummarizer:
#     """Async video summarizer that handles multiple concurrent requests"""
    
#     def __init__(self, azure_openai_key: str, azure_openai_endpoint: str, 
#                  azure_api_version: str, azure_deployment_name: str, 
#                  murf_api_key: str = None, tts_engine: str = 'gtts', 
#                  target_wpm: float = 160.0):
        
#         # Initialize Azure OpenAI Client
#         self.openai_client = openai.AzureOpenAI(
#             api_key=azure_openai_key,
#             azure_endpoint=azure_openai_endpoint,
#             api_version=azure_api_version
#         )
#         self.azure_deployment_name = azure_deployment_name
        
#         self.tts_engine = tts_engine.lower()
        
#         if self.tts_engine == 'murf':
#             if not murf_api_key:
#                 raise ValueError("Murf API key is required when using the 'murf' TTS engine.")
#             self.murf_client = Murf(api_key=murf_api_key)
#         elif self.tts_engine == 'gtts':
#             self.murf_client = None
#             logger.info("Using gTTS for audio generation.")
#         else:
#             raise ValueError(f"Unsupported TTS engine: {tts_engine}. Choose 'murf' or 'gtts'.")
        
#         self.target_wpm = target_wpm
        
#         # Cost tracking
#         self.total_cost = 0.0
#         self.total_prompt_tokens = 0
#         self.total_completion_tokens = 0
        
#         # Thread pool for CPU-intensive tasks
#         self.executor = ThreadPoolExecutor(max_workers=4)
    
#     def _update_usage_and_cost(self, usage, model: str = "gpt-4o-mini"):
#         """Calculates cost for a single API call and updates totals."""
#         if not usage:
#             logger.warning("No usage data found in OpenAI response.")
#             return
        
#         prompt_tokens = usage.prompt_tokens
#         completion_tokens = usage.completion_tokens
        
#         model_rates = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
#         prompt_cost = (prompt_tokens / 1_000_000) * model_rates["prompt_per_million"]
#         completion_cost = (completion_tokens / 1_000_000) * model_rates["completion_per_million"]
#         call_cost = prompt_cost + completion_cost
        
#         self.total_prompt_tokens += prompt_tokens
#         self.total_completion_tokens += completion_tokens
#         self.total_cost += call_cost
        
#         logger.info(f"API Call Cost: ${call_cost:.6f} | Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
#     def parse_srt_content(self, srt_content: str) -> List[SRTSegment]:
#         """Parse SRT content string into segments"""
#         logger.info("Parsing SRT content...")
        
#         blocks = re.split(r'\n\s*\n', srt_content.strip())
#         segments = []
        
#         for block in blocks:
#             if not block.strip():
#                 continue
#             lines = block.strip().split('\n')
#             if len(lines) < 3:
#                 continue
            
#             try:
#                 index = int(lines[0])
#                 time_line = lines[1]
#                 text = ' '.join(lines[2:]).strip()
                
#                 time_match = re.match(
#                     r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', 
#                     time_line
#                 )
#                 if not time_match:
#                     continue
                
#                 start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, time_match.groups())
#                 start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
#                 end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
#                 duration = end_seconds - start_seconds
#                 word_count = len(text.split())
#                 words_per_second = word_count / duration if duration > 0 else 0
                
#                 segment = SRTSegment(
#                     index=index,
#                     start_time=f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}",
#                     end_time=f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}",
#                     start_seconds=start_seconds,
#                     end_seconds=end_seconds,
#                     duration=duration,
#                     text=text,
#                     word_count=word_count,
#                     words_per_second=words_per_second
#                 )
#                 segments.append(segment)
                
#             except (ValueError, IndexError) as e:
#                 logger.warning(f"Skipping malformed segment: {e}")
#                 continue
        
#         logger.info(f"Parsed {len(segments)} segments")
#         return segments
    
#     async def analyze_video_structure_async(self, segments: List[SRTSegment]) -> Dict:
#         """Async analysis of video structure and themes using AI"""
#         logger.info("Analyzing video structure and themes...")
#         full_text = " ".join([s.text for s in segments])
        
#         structure_prompt = f"""
#         Analyze the following video transcript.
#         Respond with a single JSON object that conforms to the following schema:
#         {{
#             "main_theme": "string",
#             "key_sections": ["string"],
#             "important_terms": ["string"],
#             "tone": "string",
#             "transition_markers": ["string"]
#         }}

#         TRANSCRIPT: {full_text}...
#         """
        
#         try:
#             # Run API call in thread pool to avoid blocking
#             loop = asyncio.get_event_loop()
            
#             def make_openai_call():
#                 return self.openai_client.chat.completions.create(
#                     model=self.azure_deployment_name,
#                     messages=[{"role": "user", "content": structure_prompt}],
#                     response_format={"type": "json_object"},
#                     temperature=0.7
#                 )
            
#             response = await loop.run_in_executor(self.executor, make_openai_call)
            
#             self._update_usage_and_cost(response.usage, model="gpt-4o-mini")
            
#             response_json = json.loads(response.choices[0].message.content)
#             analysis = VideoAnalysis(**response_json)
            
#             analysis_dict = analysis.model_dump()
#             logger.info("Video structure analysis complete")
#             return analysis_dict
            
#         except Exception as e:
#             logger.error(f"Error in structure analysis: {e}. Using fallback.")
#             fallback_analysis = {
#                 "main_theme": "Technical demonstration",
#                 "key_sections": ["introduction", "demonstration", "features"],
#                 "important_terms": [],
#                 "tone": "instructional",
#                 "transition_markers": ["so", "now", "let's", "next"]
#             }
#             return fallback_analysis

#     async def optimize_segment_with_context_async(self, segment: SRTSegment, video_context: Dict, 
#                                                   prev_optimized: List[OptimizedSegment] = None, 
#                                                   upcoming_segments: List[SRTSegment] = None) -> OptimizedSegment:
#         """Async optimization of a single segment with context awareness"""
#         prev_optimized = prev_optimized or []
#         upcoming_segments = upcoming_segments or []
        
#         json_schema = SegmentOptimization.model_json_schema()
        
#         optimization_prompt = f"""
#         You are an expert video script editor. Your task is to optimize the 'CURRENT SEGMENT' for clarity and brevity while maintaining the original meaning and tone.
        
#         Respond with a single JSON object matching this schema:
#         {json.dumps(json_schema, indent=2)}

#         CONTEXT:
#         - Video Theme: {video_context.get("main_theme", "N/A")}
#         - Important Terms: {video_context.get("important_terms", [])}
#         - Last sentence was: {' '.join(s.optimized_text for s in prev_optimized[-2:])}
#         - Next sentence will be: {' '.join(s.text for s in upcoming_segments[:2])}

#         CURRENT SEGMENT TO OPTIMIZE: "{segment.text}"
#         """
        
#         try:
#             # Run API call in thread pool
#             loop = asyncio.get_event_loop()
            
#             def make_openai_call():
#                 return self.openai_client.chat.completions.create(
#                     model=self.azure_deployment_name,
#                     messages=[{"role": "user", "content": optimization_prompt}],
#                     response_format={"type": "json_object"},
#                     temperature=0.2
#                 )
            
#             response = await loop.run_in_executor(self.executor, make_openai_call)
            
#             self._update_usage_and_cost(response.usage, model="gpt-4o-mini")
            
#             response_json = json.loads(response.choices[0].message.content)
#             result = SegmentOptimization(**response_json)
            
#             optimized_text = result.optimized_text
#             optimized_word_count = len(optimized_text.split())
#             estimated_duration = (optimized_word_count / self.target_wpm) * 60
#             speed_multiplier = segment.duration / estimated_duration if estimated_duration > 0 else 1.0
            
#             return OptimizedSegment(
#                 original=segment,
#                 optimized_text=optimized_text,
#                 optimized_word_count=optimized_word_count,
#                 estimated_speech_duration=estimated_duration,
#                 speed_multiplier=max(0.5, min(4.0, speed_multiplier)),
#                 reasoning=result.reasoning
#             )
            
#         except Exception as e:
#             logger.error(f"Error optimizing segment {segment.index}: {e}")
#             return OptimizedSegment(
#                 original=segment,
#                 optimized_text=segment.text,
#                 optimized_word_count=segment.word_count,
#                 estimated_speech_duration=segment.duration,
#                 speed_multiplier=1.0,
#                 reasoning="Fallback: no optimization applied due to error"
#             )
    
#     async def optimize_all_segments_async(self, segments: List[SRTSegment]) -> List[OptimizedSegment]:
#         """Async optimization of all segments with context awareness"""
#         logger.info("Starting context-aware optimization of all segments...")
        
#         # Analyze video structure first
#         video_context = await self.analyze_video_structure_async(segments)
        
#         optimized_segments = []
#         context_window = 6
        
#         # Process segments with limited concurrency to avoid overwhelming the API
#         semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent API calls
        
#         async def optimize_single_segment(i):
#             async with semaphore:
#                 current_segment = segments[i]
#                 prev_optimized = optimized_segments[max(0, i - context_window):] if optimized_segments else []
#                 upcoming_segments = segments[i + 1:min(len(segments), i + context_window + 1)]
                
#                 optimized = await self.optimize_segment_with_context_async(
#                     current_segment, video_context, prev_optimized, upcoming_segments
#                 )
#                 return i, optimized
        
#         # Process segments in batches to maintain order
#         batch_size = 5
#         for batch_start in range(0, len(segments), batch_size):
#             batch_end = min(batch_start + batch_size, len(segments))
#             batch_tasks = [optimize_single_segment(i) for i in range(batch_start, batch_end)]
            
#             batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
#             # Sort results by index and add to optimized_segments
#             batch_results = [(i, result) for i, result in batch_results if not isinstance(result, Exception)]
#             batch_results.sort(key=lambda x: x[0])
            
#             for _, optimized in batch_results:
#                 optimized_segments.append(optimized)
            
#             logger.info(f"Optimized {len(optimized_segments)}/{len(segments)} segments")
        
#         logger.info("Optimization complete.")
#         return optimized_segments
    
#     async def generate_tts_audio_async(self, text: str, temp_dir: str, voice_id: str = 'en-IN-arohi', 
#                                        retries: int = 3, delay: int = 5) -> tuple:
#         """
#         Async generation of TTS audio to temporary file
#         Returns: (temp_file_path, duration)
#         """
#         if not text.strip():
#             logger.warning("Empty text provided for TTS, skipping.")
#             return None, 0.0
        
#         audio_filename = "tts_audio.wav"
#         output_path = os.path.join(temp_dir, audio_filename)
        
#         if self.tts_engine == 'murf':
#             # Murf TTS Logic - run in thread pool
#             for attempt in range(retries):
#                 try:
#                     logger.info(f"Generating Murf TTS for: '{text[:40]}...'")
                    
#                     loop = asyncio.get_event_loop()
                    
#                     def generate_murf_tts():
#                         res = self.murf_client.text_to_speech.generate(
#                             text=text,
#                             voice_id=voice_id,
#                         )
#                         return res.audio_file, res.audio_length_in_seconds
                    
#                     audio_url, audio_duration = await loop.run_in_executor(self.executor, generate_murf_tts)
                    
#                     if not audio_url:
#                         logger.error("Murf API did not return an audio URL.")
#                         return None, 0.0
                    
#                     # Download audio file asynchronously
#                     async with aiohttp.ClientSession() as session:
#                         async with session.get(audio_url) as response:
#                             response.raise_for_status()
#                             with open(output_path, 'wb') as f:
#                                 async for chunk in response.content.iter_chunked(8192):
#                                     f.write(chunk)
                    
#                     logger.info(f"Successfully saved Murf TTS audio to {output_path}")
#                     return output_path, audio_duration
                    
#                 except Exception as e:
#                     logger.warning(f"Error on attempt {attempt + 1}/{retries} for Murf TTS: {e}")
#                     if attempt < retries - 1:
#                         logger.info(f"Retrying in {delay} seconds...")
#                         await asyncio.sleep(delay)
#                     else:
#                         logger.error(f"Murf TTS generation failed after {retries} attempts")
#                         return None, 0.0
#             return None, 0.0
        
#         elif self.tts_engine == 'gtts':
#             # gTTS Logic - run in thread pool
#             try:
#                 logger.info(f"Generating gTTS for: '{text[:40]}...'")
                
#                 loop = asyncio.get_event_loop()
                
#                 def generate_gtts():
#                     tts = gTTS(text=text, lang='en')
#                     tts.save(output_path)
                
#                 await loop.run_in_executor(self.executor, generate_gtts)
                
#                 if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                     duration = await self._get_media_duration_async(output_path)
#                     logger.info(f"Successfully saved gTTS audio to {output_path} (Duration: {duration:.2f}s)")
#                     return output_path, duration
#                 else:
#                     logger.error(f"gTTS failed to create a valid audio file")
#                     return None, 0.0
                    
#             except Exception as e:
#                 logger.error(f"gTTS generation failed: {e}")
#                 return None, 0.0
        
#         else:
#             logger.error(f"TTS generation skipped: unknown engine '{self.tts_engine}'")
#             return None, 0.0
    
#     async def _get_media_duration_async(self, file_path: str) -> float:
#         """Get media file duration using sync ffprobe in thread pool"""
#         try:
#             import subprocess
#             loop = asyncio.get_event_loop()
            
#             def get_duration_sync():
#                 result = subprocess.run([
#                     'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
#                     '-of', 'default=noprint_wrappers=1:nokey=1', file_path
#                 ], capture_output=True, text=True, timeout=30)
                
#                 if result.returncode == 0 and result.stdout.strip():
#                     return float(result.stdout.strip())
#                 else:
#                     logger.error(f"ffprobe failed for {file_path}")
#                     logger.error(f"Return code: {result.returncode}")
#                     logger.error(f"stderr: {result.stderr}")
#                     return 0.0
            
#             duration = await loop.run_in_executor(self.executor, get_duration_sync)
#             if duration > 0:
#                 logger.info(f"Duration for {file_path}: {duration:.2f}s")
#             return duration
            
#         except Exception as e:
#             logger.error(f"Error getting duration for {file_path}: {e}")
#             return 0.0

#     def _create_rounded_mask_image(self, width: int, height: int, radius: int, output_path: str):
#         """Creates a black PNG with a white rounded rectangle to use as a video mask."""
#         try:
#             # Create a black background image with a transparent alpha channel
#             mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
#             draw = ImageDraw.Draw(mask)
            
#             # Draw a white rounded rectangle on the mask.
#             # The alpha channel of this shape will define the video's shape.
#             draw.rounded_rectangle(
#                 (0, 0, width, height),
#                 fill=(255, 255, 255, 255), # White and fully opaque
#                 radius=radius
#             )
#             mask.save(output_path, 'PNG')
#             logger.info(f"Successfully created temporary mask at {output_path}")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to create mask image: {e}")
#             return False

#     async def _apply_background_async(self, input_video_path: str, background_image_path: str,
#                                       output_video_path: str, overlay_options: dict):
#         """
#         Overlays the video onto a background using a robust PNG mask for rounded corners.
#         This version adds a crop filter to ensure final dimensions are even, fixing encoder errors.
#         """
#         defaults = {
#             'width': 1280, 'height': 720, 'x': '(main_w-overlay_w)/2',
#             'y': '(main_h-overlay_h)/2', 'corner_radius': 0
#         }
#         opts = {**defaults, **overlay_options}
#         logger.info(f"Applying background with options: {opts}")

#         mask_path = None
#         temp_mask_file = None
#         try:
#             # Crop the background to ensure its dimensions are even numbers.
#             filter_chain_prefix = "[0:v]crop=floor(iw/2)*2:floor(ih/2)*2[bg];"

#             if opts['corner_radius'] > 0:
#                 temp_mask_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#                 mask_path = temp_mask_file.name
#                 temp_mask_file.close()

#                 if not self._create_rounded_mask_image(opts['width'], opts['height'], opts['corner_radius'], mask_path):
#                     raise ValueError("Mask image creation failed.")

#                 filter_chain = (
#                     filter_chain_prefix +
#                     f"[1:v]scale={opts['width']}:{opts['height']}[scaled_video];"
#                     f"[scaled_video][2:v]alphamerge[masked_video];"
#                     f"[bg][masked_video]overlay={opts['x']}:{opts['y']}[v_out]"
#                 )
                
#                 cmd = [
#                     'ffmpeg', '-y',
#                     '-loop', '1', '-i', background_image_path,
#                     '-i', input_video_path,
#                     '-i', mask_path,
#                     '-filter_complex', filter_chain,
#                     '-map', '[v_out]', '-map', '1:a',
#                     '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
#                     '-pix_fmt', 'yuv420p', '-shortest',
#                     output_video_path
#                 ]
#             else:
#                 filter_chain = (
#                     filter_chain_prefix +
#                     f"[1:v]scale={opts['width']}:{opts['height']}[scaled_video];"
#                     f"[bg][scaled_video]overlay={opts['x']}:{opts['y']}[v_out]"
#                 )
#                 cmd = [
#                     'ffmpeg', '-y',
#                     '-loop', '1', '-i', background_image_path,
#                     '-i', input_video_path,
#                     '-filter_complex', filter_chain,
#                     '-map', '[v_out]', '-map', '1:a',
#                     '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
#                     '-pix_fmt', 'yuv420p', '-shortest',
#                     output_video_path
#                 ]

#             def apply_bg_sync():
#                 result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
#                 return result.returncode, result.stderr

#             loop = asyncio.get_event_loop()
#             returncode, stderr = await loop.run_in_executor(self.executor, apply_bg_sync)

#             if returncode == 0 and os.path.exists(output_video_path):
#                 logger.info(f"Successfully applied framed background to {output_video_path}")
#                 return True
#             else:
#                 logger.error(f"Failed to apply frame. FFmpeg returned {returncode}")
#                 logger.error(f"FFmpeg command: {' '.join(cmd)}")
#                 logger.error(f"FFmpeg stderr: {stderr}")
#                 return False
#         except Exception as e:
#             logger.error(f"An exception occurred while applying frame: {e}")
#             return False
#         finally:
#             if mask_path and os.path.exists(mask_path):
#                 os.remove(mask_path)
#                 logger.info(f"Cleaned up temporary mask file: {mask_path}")

#     # <<< START OF REFACTORED METHOD >>>
#     async def process_video_segments_async(self, input_video: str, optimized_segments: List[OptimizedSegment]) -> str:
#         """
#         Async processing of video segments in a two-stage process.
#         Stage 1: Generate all audio concurrently.
#         Stage 2: Process all video segments concurrently.
#         """
#         with tempfile.TemporaryDirectory() as temp_dir:
#             logger.info(f"Processing video with {len(optimized_segments)} segments in {temp_dir}")

#             # --- PRE-FLIGHT CHECK ---
#             video_duration = await self._get_media_duration_async(input_video)
#             logger.info(f"Input video duration check: {video_duration:.2f}s")
#             if video_duration == 0.0:
#                 if not os.path.exists(input_video):
#                     raise ValueError(f"Input video file does not exist: {input_video}")
#                 raise ValueError(f"Could not read video duration. File may be corrupted: {input_video}")

#             # --- STAGE 1: GENERATE ALL AUDIO FILES CONCURRENTLY ---
#             logger.info("--- Stage 1: Generating all audio files ---")
#             audio_tasks = []
#             for i, opt_seg in enumerate(optimized_segments):
#                 segment_temp_dir = os.path.join(temp_dir, f"segment_{i}")
#                 os.makedirs(segment_temp_dir, exist_ok=True)
#                 audio_tasks.append(
#                     self.generate_tts_audio_async(opt_seg.optimized_text, segment_temp_dir)
#                 )
            
#             audio_generation_results = await asyncio.gather(*audio_tasks, return_exceptions=True)

#             # Collect successful audio results to pass to the next stage
#             successful_audio_data = []
#             for i, result in enumerate(audio_generation_results):
#                 if isinstance(result, Exception) or not result or result[0] is None or result[1] <= 0:
#                     logger.warning(f"Skipping segment {i} due to audio generation failure.")
#                     continue
#                 audio_path, audio_duration = result
#                 successful_audio_data.append({
#                     "index": i,
#                     "opt_seg": optimized_segments[i],
#                     "audio_path": audio_path,
#                     "audio_duration": audio_duration
#                 })

#             if not successful_audio_data:
#                 raise ValueError("All audio generation tasks failed. Cannot proceed.")
#             logger.info(f"--- Stage 1 Complete: Successfully generated {len(successful_audio_data)} audio files ---")


#             # --- STAGE 2: PROCESS ALL VIDEO SEGMENTS CONCURRENTLY ---
#             logger.info("--- Stage 2: Processing all video segments using generated audio data ---")
#             semaphore = asyncio.Semaphore(2)  # Limit concurrent FFmpeg processes

#             async def process_single_video_segment(segment_data: dict):
#                 async with semaphore:
#                     i = segment_data["index"]
#                     opt_seg = segment_data["opt_seg"]
#                     actual_audio_duration = segment_data["audio_duration"]
#                     segment_temp_dir = os.path.dirname(segment_data["audio_path"])

#                     segment_video_path = os.path.join(segment_temp_dir, f"video_{i:04d}.mp4")
#                     start_time = opt_seg.original.start_seconds
#                     original_video_seg_duration = opt_seg.original.duration
#                     speed_multiplier = original_video_seg_duration / actual_audio_duration

#                     extract_cmd = [
#                         'ffmpeg', '-y', '-ss', f'{start_time:.3f}', '-i', input_video,
#                         '-t', f'{original_video_seg_duration:.3f}',
#                         '-filter:v', f'setpts={1/speed_multiplier:.4f}*PTS',
#                         '-c:v', 'libx264', '-crf', '23', '-an', segment_video_path
#                     ]

#                     try:
#                         def extract_video_sync():
#                             result = subprocess.run(
#                                 extract_cmd, capture_output=True, text=True, timeout=120
#                             )
#                             return result.returncode, result.stdout, result.stderr

#                         loop = asyncio.get_event_loop()
#                         returncode, _, stderr_sync = await loop.run_in_executor(
#                             self.executor, extract_video_sync
#                         )

#                         if returncode == 0 and os.path.exists(segment_video_path) and os.path.getsize(segment_video_path) > 0:
#                             logger.info(f"Segment {i+1}: Speed {speed_multiplier:.2f}x, Duration: {actual_audio_duration:.2f}s")
#                             return segment_video_path, segment_data["audio_path"]
#                         else:
#                             logger.error(f"FFmpeg failed for segment {i} (code {returncode}): {stderr_sync}")
#                             return None
#                     except Exception as e:
#                         logger.error(f"Exception processing video segment {i}: {e}")
#                         return None

#             video_tasks = [process_single_video_segment(data) for data in successful_audio_data]
#             video_processing_results = await asyncio.gather(*video_tasks, return_exceptions=True)
#             logger.info("--- Stage 2 Complete: Video segment processing finished ---")
            
#             # --- STAGE 3: COLLECT RESULTS AND COMBINE ---
#             video_segments = []
#             audio_segments = []
#             for result in video_processing_results:
#                 if result and not isinstance(result, Exception):
#                     video_path, audio_path = result
#                     video_segments.append(video_path)
#                     audio_segments.append(audio_path)
            
#             if not video_segments:
#                 raise ValueError("No valid video segments were created after processing stage.")

#             # Combine all processed segments into a temporary video file
#             internal_output_video = os.path.join(temp_dir, "final_output.mp4")
#             await self._robust_combine_segments_async(video_segments, audio_segments, internal_output_video, temp_dir)
            
#             # Move the final video to a persistent path that won't be auto-deleted
#             with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
#                 persistent_output_path = dest_f.name
#             shutil.move(internal_output_video, persistent_output_path)
            
#             logger.info(f"Video processing complete. Final output at: {persistent_output_path}")
#             return persistent_output_path
            
#     # <<< END OF REFACTORED METHOD >>>

#     async def _robust_combine_segments_async(self, video_segments: List[str], audio_segments: List[str], 
#                                             output_video: str, temp_dir: str):
#         """Async combination of video and audio segments using robust concatenation method"""
#         if len(video_segments) != len(audio_segments):
#             raise ValueError(f"Mismatch: {len(video_segments)} video vs {len(audio_segments)} audio segments")
        
#         logger.info(f"Combining {len(video_segments)} segments with audio-priority timing...")
        
#         combined_segments_for_concat = []
        
#         # Stage 1: Create combined segments with audio
#         combine_tasks = []
#         for i, (video_seg_path, audio_seg_path) in enumerate(zip(video_segments, audio_segments)):
#             combine_tasks.append(self._combine_single_segment_async(i, video_seg_path, audio_seg_path, temp_dir))
        
#         combine_results = await asyncio.gather(*combine_tasks, return_exceptions=True)
        
#         for result in combine_results:
#             if result and not isinstance(result, Exception):
#                 combined_segments_for_concat.append(Path(result))
        
#         if not combined_segments_for_concat:
#             raise ValueError("No segments were successfully combined.")
        
#         # Stage 2: Use robust concatenation method
#         temp_ts_dir = Path(temp_dir) / "temp_ts_files"
#         temp_ts_dir.mkdir(exist_ok=True)
        
#         try:
#             # Sort the combined segments naturally
#             combined_segments_for_concat.sort(key=self._natural_sort_key)
            
#             # Sanitize clips to TS format
#             clean_ts_files = await self._sanitize_clips_to_ts_async(combined_segments_for_concat, temp_ts_dir)
            
#             if not clean_ts_files or len(clean_ts_files) != len(combined_segments_for_concat):
#                 logger.error("Aborting due to failure in sanitization stage.")
#                 raise ValueError("Failed to sanitize all video segments")
            
#             # Concatenate TS files
#             success = await self._concatenate_ts_files_async(clean_ts_files, output_video, temp_dir)
            
#             if not success:
#                 raise ValueError("Failed to concatenate segments")
            
#             logger.info("Successfully combined all segments using robust method.")
            
#         finally:
#             # Cleanup handled automatically by TemporaryDirectory
#             pass
    
#     async def _combine_single_segment_async(self, i: int, video_seg_path: str, audio_seg_path: str, temp_dir: str) -> str:
#         """Combine single video and audio segment using sync FFmpeg"""
#         final_segment_path = os.path.join(temp_dir, f"combined_{i:04d}.mp4")
        
#         audio_dur = await self._get_media_duration_async(audio_seg_path)
#         if audio_dur == 0.0:
#             logger.warning(f"Skipping segment {i} due to zero duration audio.")
#             return None
        
#         combine_cmd = [
#             'ffmpeg', '-y',
#             '-i', video_seg_path,
#             '-i', audio_seg_path,
#             '-c:v', 'copy',
#             '-c:a', 'aac', '-b:a', '192k',
#             '-t', f'{audio_dur:.5f}',
#             final_segment_path
#         ]
        
#         try:
#             import subprocess
            
#             def combine_sync():
#                 result = subprocess.run(
#                     combine_cmd,
#                     capture_output=True,
#                     text=True,
#                     timeout=60  # 1 minute timeout
#                 )
#                 return result.returncode, result.stdout, result.stderr
            
#             loop = asyncio.get_event_loop()
#             returncode, stdout, stderr = await loop.run_in_executor(self.executor, combine_sync)
            
#             if returncode == 0 and os.path.exists(final_segment_path) and os.path.getsize(final_segment_path) > 0:
#                 logger.info(f"Combined segment {i} successfully")
#                 return final_segment_path
#             else:
#                 logger.error(f"Error combining segment {i}: return code {returncode}")
#                 logger.error(f"stderr: {stderr}")
#                 return None
                
#         except Exception as e:
#             logger.error(f"Exception combining segment {i}: {e}")
#             return None
    
#     def _natural_sort_key(self, filename: Path) -> int:
#         """Natural sorting key for combined video files"""
#         match = re.search(r'combined_(\d+)\.mp4', filename.name)
#         if match:
#             return int(match.group(1))
#         return 0
    
#     async def _sanitize_clips_to_ts_async(self, video_files: List[Path], temp_ts_dir: Path) -> List[Path]:
#         """Re-encode clips to clean MPEG-TS format using sync FFmpeg"""
#         ts_files = []
#         logger.info(f"Sanitizing {len(video_files)} clips to TS format")
        
#         # Process files with limited concurrency
#         semaphore = asyncio.Semaphore(2)
        
#         async def sanitize_single_file(i, video_file):
#             async with semaphore:
#                 logger.info(f"Sanitizing file {i+1}/{len(video_files)}: {video_file.name}")
#                 output_ts_path = temp_ts_dir / f"clean_{i:04d}.ts"
                
#                 cmd = [
#                     'ffmpeg', '-y', '-i', str(video_file.absolute()),
#                     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
#                     '-vf', 'fps=30,format=yuv420p',
#                     '-c:a', 'aac', '-b:a', '192k',
#                     str(output_ts_path)
#                 ]
                
#                 try:
#                     import subprocess
                    
#                     def sanitize_sync():
#                         result = subprocess.run(
#                             cmd,
#                             capture_output=True,
#                             text=True,
#                             timeout=120
#                         )
#                         return result.returncode, result.stdout, result.stderr
                    
#                     loop = asyncio.get_event_loop()
#                     returncode, stdout, stderr = await loop.run_in_executor(self.executor, sanitize_sync)
                    
#                     if returncode == 0 and os.path.exists(output_ts_path) and os.path.getsize(output_ts_path) > 0:
#                         logger.info(f"Sanitized {video_file.name} successfully")
#                         return output_ts_path
#                     else:
#                         logger.error(f"Failed to sanitize {video_file.name}")
#                         logger.error(f"Return code: {returncode}")
#                         logger.error(f"stderr: {stderr}")
#                         return None
                        
#                 except Exception as e:
#                     logger.error(f"Error sanitizing {video_file.name}: {e}")
#                     return None
        
#         tasks = [sanitize_single_file(i, video_file) for i, video_file in enumerate(video_files)]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         for result in results:
#             if result and not isinstance(result, Exception):
#                 ts_files.append(result)
        
#         return ts_files
    
#     async def _concatenate_ts_files_async(self, ts_files: List[Path], output_video: str, temp_dir: str) -> bool:
#             """Async concatenation of clean TS files into final MP4 using a robust sync method"""
#             logger.info(f"Concatenating {len(ts_files)} clean clips into {output_video}")
            
#             # Create temporary file list for ffmpeg's concat demuxer
#             list_path = Path(temp_dir) / "concat_list.txt"
#             with open(list_path, 'w', encoding='utf-8') as f:
#                 for ts_file in ts_files:
#                     # Use as_posix() for cross-platform path compatibility in the list file
#                     f.write(f"file '{ts_file.absolute().as_posix()}'\n")
            
#             cmd = [
#                 'ffmpeg', '-y', 
#                 '-f', 'concat', 
#                 '-safe', '0',         # Needed when using absolute paths in the list
#                 '-i', str(list_path),
#                 '-c', 'copy',        # Copy codecs without re-encoding for speed
#                 output_video
#             ]
            
#             try:
#                 import subprocess
    
#                 def concat_sync():
#                     """Wrapper for the synchronous subprocess call."""
#                     # Use a longer timeout as final concatenation can take time
#                     result = subprocess.run(
#                         cmd,
#                         capture_output=True,
#                         text=True,
#                         timeout=300  # 5-minute timeout for the final combine step
#                     )
#                     return result.returncode, result.stderr
    
#                 loop = asyncio.get_event_loop()
#                 returncode, stderr = await loop.run_in_executor(self.executor, concat_sync)
                
#                 if returncode == 0 and os.path.exists(output_video) and os.path.getsize(output_video) > 0:
#                     logger.info("Final concatenation successful!")
#                     return True
#                 else:
#                     logger.error(f"Final concatenation failed with return code {returncode}.")
#                     logger.error(f"FFmpeg stderr: {stderr}")
#                     return False
                    
#             except subprocess.TimeoutExpired:
#                 logger.error("Final concatenation timed out after 5 minutes.")
#                 return False
#             except Exception as e:
#                 logger.error(f"An unexpected error occurred during concatenation: {e}")
#                 return False
    
#     def get_cost_summary(self) -> Dict:
#         """Return cost and usage summary"""
#         total_tokens = self.total_prompt_tokens + self.total_completion_tokens
#         return {
#             "total_prompt_tokens": self.total_prompt_tokens,
#             "total_completion_tokens": self.total_completion_tokens,
#             "total_tokens": total_tokens,
#             "estimated_cost_usd": self.total_cost
#         }

#     async def process_srt_content_to_video_async(
#         self,
#         srt_content: str,
#         input_video: str,
#         background_image_path: str = None,
#         overlay_options: Dict = None,
#     ) -> tuple:
#         """
#         Main async method to process SRT content and create optimized video.
#         Optionally applies a background image/frame to the final video.
#         """
#         logger.info("Starting SRT content processing for video optimization")
    
#         try:
#             segments = self.parse_srt_content(srt_content)
#             if not segments:
#                 raise ValueError("No valid segments found in SRT content")
    
#             optimized_segments = await self.optimize_all_segments_async(segments)
    
#             intermediate_video_path = await self.process_video_segments_async(
#                 input_video, optimized_segments
#             )
    
#             final_video_path = intermediate_video_path
#             print(f"this si the fnal video path{final_video_path}")
    
#             if background_image_path:
#                 print(f"this is the backgorundimage path{background_image_path}")
#                 if not os.path.exists(background_image_path):
#                     logger.error(
#                         f"Background image not found at {background_image_path}. Skipping."
#                     )
#                 else:
#                     with tempfile.NamedTemporaryFile(
#                         suffix=".mp4", delete=False
#                     ) as dest_f:
#                         composited_video_path = dest_f.name
    
#                     # Use provided overlay options, or an empty dict for defaults
#                     current_overlay_options = overlay_options or {}
    
#                     success = await self._apply_background_async(
#                         intermediate_video_path,
#                         background_image_path,
#                         composited_video_path,
#                         current_overlay_options,
#                     )
    
#                     if success:
#                         final_video_path = composited_video_path
#                         os.remove(intermediate_video_path)
#                     else:
#                         logger.warning(
#                             "Could not apply frame. Returning the original video."
#                         )
#                         os.remove(composited_video_path)
    
#             cost_summary = self.get_cost_summary()
    
#             logger.info("SRT processing and video optimization complete")
#             return final_video_path, cost_summary
    
#         except Exception as e:
#             logger.error(f"Error in SRT processing: {e}")
#             raise

#     def __del__(self):
#         """Cleanup thread pool executor"""
#         if hasattr(self, 'executor'):
#             self.executor.shutdown(wait=False)

import re
import json
import os
import logging
import time
import asyncio
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import openai
import subprocess
from pydantic import BaseModel
import requests
import aiohttp
from requests.exceptions import RequestException
from murf import Murf
from gtts import gTTS
import tempfile
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# OpenAI Model Pricing (USD per 1 million tokens)
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt_per_million": 0.15, "completion_per_million": 0.6},
}

# Pydantic models for OpenAI structured outputs
class VideoAnalysis(BaseModel):
    """Structured model for video analysis response"""
    main_theme: str
    key_sections: List[str]
    important_terms: List[str]
    tone: str
    transition_markers: List[str]

class SegmentOptimization(BaseModel):
    """Structured model for segment optimization response"""
    optimized_text: str
    reasoning: str
    confidence: float
    preserved_terms: List[str]

@dataclass
class SRTSegment:
    """Represents a single SRT subtitle segment"""
    index: int
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    duration: float
    text: str
    word_count: int
    words_per_second: float

@dataclass
class OptimizedSegment:
    """Represents an optimized segment with new script and timing"""
    original: SRTSegment
    optimized_text: str
    optimized_word_count: int
    estimated_speech_duration: float
    speed_multiplier: float
    reasoning: str


class FFmpegProcessor:
    """
    Handles all FFmpeg and ffprobe subprocess executions asynchronously.
    All methods run synchronous subprocess calls in a thread pool to avoid blocking the event loop.
    """
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor

    async def get_media_duration(self, file_path: str) -> float:
        """Get media file duration using ffprobe."""
        try:
            loop = asyncio.get_event_loop()
            
            def get_duration_sync():
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', file_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
                else:
                    logger.error(f"ffprobe failed for {file_path}")
                    logger.error(f"Return code: {result.returncode}")
                    logger.error(f"stderr: {result.stderr}")
                    return 0.0
            
            duration = await loop.run_in_executor(self.executor, get_duration_sync)
            if duration > 0:
                logger.info(f"Duration for {file_path}: {duration:.2f}s")
            return duration
            
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {e}")
            return 0.0

    async def apply_background(
        self, input_video_path: str, background_image_path: str,
        output_video_path: str, overlay_options: dict, mask_path: str = None
    ) -> bool:
        """Applies a background to a video using FFmpeg."""
        defaults = {
            'width': 1280, 'height': 720, 'x': '(main_w-overlay_w)/2',
            'y': '(main_h-overlay_h)/2', 'corner_radius': 0
        }
        opts = {**defaults, **overlay_options}
        
        filter_chain_prefix = "[0:v]crop=floor(iw/2)*2:floor(ih/2)*2[bg];"

        if mask_path:
            filter_chain = (
                filter_chain_prefix +
                f"[1:v]scale={opts['width']}:{opts['height']}[scaled_video];"
                f"[scaled_video][2:v]alphamerge[masked_video];"
                f"[bg][masked_video]overlay={opts['x']}:{opts['y']}[v_out]"
            )
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1', '-i', background_image_path,
                '-i', input_video_path,
                '-i', mask_path,
                '-filter_complex', filter_chain,
                '-map', '[v_out]', '-map', '1:a',
                '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
                '-pix_fmt', 'yuv420p', '-shortest',
                output_video_path
            ]
        else:
            filter_chain = (
                filter_chain_prefix +
                f"[1:v]scale={opts['width']}:{opts['height']}[scaled_video];"
                f"[bg][scaled_video]overlay={opts['x']}:{opts['y']}[v_out]"
            )
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1', '-i', background_image_path,
                '-i', input_video_path,
                '-filter_complex', filter_chain,
                '-map', '[v_out]', '-map', '1:a',
                '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
                '-pix_fmt', 'yuv420p', '-shortest',
                output_video_path
            ]

        def apply_bg_sync():
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return result.returncode, result.stderr

        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(self.executor, apply_bg_sync)

        if returncode == 0 and os.path.exists(output_video_path):
            logger.info(f"Successfully applied framed background to {output_video_path}")
            return True
        else:
            logger.error(f"Failed to apply frame. FFmpeg returned {returncode}")
            logger.error(f"FFmpeg command: {' '.join(cmd)}")
            logger.error(f"FFmpeg stderr: {stderr}")
            return False
            
    async def extract_and_retime_segment(
        self, input_video: str, output_path: str, start_time: float,
        original_duration: float, speed_multiplier: float
    ) -> bool:
        """Extracts and retimes a single video segment."""
        extract_cmd = [
            'ffmpeg', '-y', '-ss', f'{start_time:.3f}', '-i', input_video,
            '-t', f'{original_duration:.3f}',
            '-filter:v', f'setpts={1/speed_multiplier:.4f}*PTS',
            '-c:v', 'libx264', '-crf', '23', '-an', output_path
        ]
        
        def extract_video_sync():
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120)
            return result.returncode, result.stderr

        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(self.executor, extract_video_sync)
        
        if returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logger.error(f"FFmpeg failed for segment extraction (code {returncode}): {stderr}")
            return False
            
    async def combine_video_audio_segment(
        self, video_path: str, audio_path: str, output_path: str, audio_duration: float
    ) -> bool:
        """Combines a single video segment with its corresponding audio segment."""
        combine_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-t', f'{audio_duration:.5f}',
            output_path
        ]
        
        def combine_sync():
            result = subprocess.run(combine_cmd, capture_output=True, text=True, timeout=60)
            return result.returncode, result.stderr
            
        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(self.executor, combine_sync)
        
        if returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logger.error(f"Error combining segment: return code {returncode}")
            logger.error(f"stderr: {stderr}")
            return False
            
    async def sanitize_clip_to_ts(self, video_file: Path, output_ts_path: Path) -> bool:
        """Re-encodes a clip to a clean MPEG-TS format."""
        cmd = [
            'ffmpeg', '-y', '-i', str(video_file.absolute()),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-vf', 'fps=30,format=yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_ts_path)
        ]
        
        def sanitize_sync():
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode, result.stderr
            
        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(self.executor, sanitize_sync)

        if returncode == 0 and os.path.exists(output_ts_path) and os.path.getsize(output_ts_path) > 0:
            return True
        else:
            logger.error(f"Failed to sanitize {video_file.name}")
            logger.error(f"Return code: {returncode}")
            logger.error(f"stderr: {stderr}")
            return False
            
    async def concatenate_ts_files(self, ts_files: List[Path], output_video: str, temp_dir: str) -> bool:
        """Concatenates a list of TS files into a final MP4."""
        list_path = Path(temp_dir) / "concat_list.txt"
        with open(list_path, 'w', encoding='utf-8') as f:
            for ts_file in ts_files:
                f.write(f"file '{ts_file.absolute().as_posix()}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_path),
            '-c', 'copy',
            output_video
        ]
        
        try:
            def concat_sync():
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                return result.returncode, result.stderr

            loop = asyncio.get_event_loop()
            returncode, stderr = await loop.run_in_executor(self.executor, concat_sync)

            if returncode == 0 and os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info("Final concatenation successful!")
                return True
            else:
                logger.error(f"Final concatenation failed with return code {returncode}.")
                logger.error(f"FFmpeg stderr: {stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Final concatenation timed out after 5 minutes.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during concatenation: {e}")
            return False


class VideoSummarizer:
    """Async video summarizer that handles multiple concurrent requests"""
    
    def __init__(self, azure_openai_key: str, azure_openai_endpoint: str, 
                 azure_api_version: str, azure_deployment_name: str, 
                 murf_api_key: str = None, tts_engine: str = 'gtts', 
                 target_wpm: float = 160.0):
        
        # Initialize Azure OpenAI Client
        self.openai_client = openai.AzureOpenAI(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_api_version
        )
        self.azure_deployment_name = azure_deployment_name
        
        self.tts_engine = tts_engine.lower()
        
        if self.tts_engine == 'murf':
            if not murf_api_key:
                raise ValueError("Murf API key is required when using the 'murf' TTS engine.")
            self.murf_client = Murf(api_key=murf_api_key)
        elif self.tts_engine == 'gtts':
            self.murf_client = None
            logger.info("Using gTTS for audio generation.")
        else:
            raise ValueError(f"Unsupported TTS engine: {tts_engine}. Choose 'murf' or 'gtts'.")
        
        self.target_wpm = target_wpm
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # Thread pool for CPU-intensive and blocking I/O tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # FFmpeg handler
        self.ffmpeg_processor = FFmpegProcessor(self.executor)
    
    def _update_usage_and_cost(self, usage, model: str = "gpt-4o-mini"):
        """Calculates cost for a single API call and updates totals."""
        if not usage:
            logger.warning("No usage data found in OpenAI response.")
            return
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
        model_rates = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        prompt_cost = (prompt_tokens / 1_000_000) * model_rates["prompt_per_million"]
        completion_cost = (completion_tokens / 1_000_000) * model_rates["completion_per_million"]
        call_cost = prompt_cost + completion_cost
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += call_cost
        
        logger.info(f"API Call Cost: ${call_cost:.6f} | Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def parse_srt_content(self, srt_content: str) -> List[SRTSegment]:
        """Parse SRT content string into segments"""
        logger.info("Parsing SRT content...")
        
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        segments = []
        
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                index = int(lines[0])
                time_line = lines[1]
                text = ' '.join(lines[2:]).strip()
                
                time_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', 
                    time_line
                )
                if not time_match:
                    continue
                
                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, time_match.groups())
                start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                duration = end_seconds - start_seconds
                word_count = len(text.split())
                words_per_second = word_count / duration if duration > 0 else 0
                
                segment = SRTSegment(
                    index=index,
                    start_time=f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}",
                    end_time=f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}",
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    duration=duration,
                    text=text,
                    word_count=word_count,
                    words_per_second=words_per_second
                )
                segments.append(segment)
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed segment: {e}")
                continue
        
        logger.info(f"Parsed {len(segments)} segments")
        return segments
    
    async def analyze_video_structure_async(self, segments: List[SRTSegment]) -> Dict:
        """Async analysis of video structure and themes using AI"""
        logger.info("Analyzing video structure and themes...")
        full_text = " ".join([s.text for s in segments])
        
        structure_prompt = f"""
        Analyze the following video transcript.
        Respond with a single JSON object that conforms to the following schema:
        {{
            "main_theme": "string",
            "key_sections": ["string"],
            "important_terms": ["string"],
            "tone": "string",
            "transition_markers": ["string"]
        }}

        TRANSCRIPT: {full_text}...
        """
        
        try:
            # Run API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def make_openai_call():
                return self.openai_client.chat.completions.create(
                    model=self.azure_deployment_name,
                    messages=[{"role": "user", "content": structure_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
            
            response = await loop.run_in_executor(self.executor, make_openai_call)
            
            self._update_usage_and_cost(response.usage, model="gpt-4o-mini")
            
            response_json = json.loads(response.choices[0].message.content)
            analysis = VideoAnalysis(**response_json)
            
            analysis_dict = analysis.model_dump()
            logger.info("Video structure analysis complete")
            return analysis_dict
            
        except Exception as e:
            logger.error(f"Error in structure analysis: {e}. Using fallback.")
            fallback_analysis = {
                "main_theme": "Technical demonstration",
                "key_sections": ["introduction", "demonstration", "features"],
                "important_terms": [],
                "tone": "instructional",
                "transition_markers": ["so", "now", "let's", "next"]
            }
            return fallback_analysis

    async def optimize_segment_with_context_async(self, segment: SRTSegment, video_context: Dict, 
                                                  prev_optimized: List[OptimizedSegment] = None, 
                                                  upcoming_segments: List[SRTSegment] = None) -> OptimizedSegment:
        """Async optimization of a single segment with context awareness"""
        prev_optimized = prev_optimized or []
        upcoming_segments = upcoming_segments or []
        
        json_schema = SegmentOptimization.model_json_schema()
        
        optimization_prompt = f"""
        You are an expert video script editor. Your task is to optimize the 'CURRENT SEGMENT' for clarity and brevity while maintaining the original meaning and tone.
        
        Respond with a single JSON object matching this schema:
        {json.dumps(json_schema, indent=2)}

        CONTEXT:
        - Video Theme: {video_context.get("main_theme", "N/A")}
        - Important Terms: {video_context.get("important_terms", [])}
        - Last sentence was: {' '.join(s.optimized_text for s in prev_optimized[-2:])}
        - Next sentence will be: {' '.join(s.text for s in upcoming_segments[:2])}

        CURRENT SEGMENT TO OPTIMIZE: "{segment.text}"
        """
        
        try:
            # Run API call in thread pool
            loop = asyncio.get_event_loop()
            
            def make_openai_call():
                return self.openai_client.chat.completions.create(
                    model=self.azure_deployment_name,
                    messages=[{"role": "user", "content": optimization_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
            
            response = await loop.run_in_executor(self.executor, make_openai_call)
            
            self._update_usage_and_cost(response.usage, model="gpt-4o-mini")
            
            response_json = json.loads(response.choices[0].message.content)
            result = SegmentOptimization(**response_json)
            
            optimized_text = result.optimized_text
            optimized_word_count = len(optimized_text.split())
            estimated_duration = (optimized_word_count / self.target_wpm) * 60
            speed_multiplier = segment.duration / estimated_duration if estimated_duration > 0 else 1.0
            
            return OptimizedSegment(
                original=segment,
                optimized_text=optimized_text,
                optimized_word_count=optimized_word_count,
                estimated_speech_duration=estimated_duration,
                speed_multiplier=max(0.5, min(4.0, speed_multiplier)),
                reasoning=result.reasoning
            )
            
        except Exception as e:
            logger.error(f"Error optimizing segment {segment.index}: {e}")
            return OptimizedSegment(
                original=segment,
                optimized_text=segment.text,
                optimized_word_count=segment.word_count,
                estimated_speech_duration=segment.duration,
                speed_multiplier=1.0,
                reasoning="Fallback: no optimization applied due to error"
            )
    
    async def optimize_all_segments_async(self, segments: List[SRTSegment]) -> List[OptimizedSegment]:
        """Async optimization of all segments with context awareness"""
        logger.info("Starting context-aware optimization of all segments...")
        
        # Analyze video structure first
        video_context = await self.analyze_video_structure_async(segments)
        
        optimized_segments = []
        context_window = 6
        
        # Process segments with limited concurrency to avoid overwhelming the API
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent API calls
        
        async def optimize_single_segment(i):
            async with semaphore:
                current_segment = segments[i]
                prev_optimized = optimized_segments[max(0, i - context_window):] if optimized_segments else []
                upcoming_segments = segments[i + 1:min(len(segments), i + context_window + 1)]
                
                optimized = await self.optimize_segment_with_context_async(
                    current_segment, video_context, prev_optimized, upcoming_segments
                )
                return i, optimized
        
        # Process segments in batches to maintain order
        batch_size = 5
        for batch_start in range(0, len(segments), batch_size):
            batch_end = min(batch_start + batch_size, len(segments))
            batch_tasks = [optimize_single_segment(i) for i in range(batch_start, batch_end)]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Sort results by index and add to optimized_segments
            batch_results = [(i, result) for i, result in batch_results if not isinstance(result, Exception)]
            batch_results.sort(key=lambda x: x[0])
            
            for _, optimized in batch_results:
                optimized_segments.append(optimized)
            
            logger.info(f"Optimized {len(optimized_segments)}/{len(segments)} segments")
        
        logger.info("Optimization complete.")
        return optimized_segments
    
    async def generate_tts_audio_async(self, text: str, temp_dir: str, voice_id: str = 'en-IN-arohi', 
                                     retries: int = 3, delay: int = 5) -> tuple:
        """
        Async generation of TTS audio to temporary file
        Returns: (temp_file_path, duration)
        """
        if not text.strip():
            logger.warning("Empty text provided for TTS, skipping.")
            return None, 0.0
        
        audio_filename = "tts_audio.wav"
        output_path = os.path.join(temp_dir, audio_filename)
        
        if self.tts_engine == 'murf':
            # Murf TTS Logic - run in thread pool
            for attempt in range(retries):
                try:
                    logger.info(f"Generating Murf TTS for: '{text[:40]}...'")
                    
                    loop = asyncio.get_event_loop()
                    
                    def generate_murf_tts():
                        res = self.murf_client.text_to_speech.generate(
                            text=text,
                            voice_id=voice_id,
                        )
                        return res.audio_file, res.audio_length_in_seconds
                    
                    audio_url, audio_duration = await loop.run_in_executor(self.executor, generate_murf_tts)
                    
                    if not audio_url:
                        logger.error("Murf API did not return an audio URL.")
                        return None, 0.0
                    
                    # Download audio file asynchronously
                    async with aiohttp.ClientSession() as session:
                        async with session.get(audio_url) as response:
                            response.raise_for_status()
                            with open(output_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                    
                    logger.info(f"Successfully saved Murf TTS audio to {output_path}")
                    return output_path, audio_duration
                    
                except Exception as e:
                    logger.warning(f"Error on attempt {attempt + 1}/{retries} for Murf TTS: {e}")
                    if attempt < retries - 1:
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Murf TTS generation failed after {retries} attempts")
                        return None, 0.0
            return None, 0.0
        
        elif self.tts_engine == 'gtts':
            # gTTS Logic - run in thread pool
            try:
                logger.info(f"Generating gTTS for: '{text[:40]}...'")
                
                loop = asyncio.get_event_loop()
                
                def generate_gtts():
                    tts = gTTS(text=text, lang='en')
                    tts.save(output_path)
                
                await loop.run_in_executor(self.executor, generate_gtts)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    duration = await self.ffmpeg_processor.get_media_duration(output_path)
                    logger.info(f"Successfully saved gTTS audio to {output_path} (Duration: {duration:.2f}s)")
                    return output_path, duration
                else:
                    logger.error(f"gTTS failed to create a valid audio file")
                    return None, 0.0
                    
            except Exception as e:
                logger.error(f"gTTS generation failed: {e}")
                return None, 0.0
        
        else:
            logger.error(f"TTS generation skipped: unknown engine '{self.tts_engine}'")
            return None, 0.0
    
    def _create_rounded_mask_image_sync(self, width: int, height: int, radius: int, output_path: str):
        """[Blocking] Creates a black PNG with a white rounded rectangle to use as a video mask."""
        try:
            # Create a black background image with a transparent alpha channel
            mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask)
            
            # Draw a white rounded rectangle on the mask.
            draw.rounded_rectangle(
                (0, 0, width, height),
                fill=(255, 255, 255, 255), # White and fully opaque
                radius=radius
            )
            mask.save(output_path, 'PNG')
            logger.info(f"Successfully created temporary mask at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create mask image: {e}")
            return False

    async def _create_rounded_mask_image_async(self, width: int, height: int, radius: int, output_path: str):
        """[Async] Wrapper to run the synchronous mask image creation in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._create_rounded_mask_image_sync, width, height, radius, output_path
        )

    async def _apply_background_async(self, input_video_path: str, background_image_path: str,
                                      output_video_path: str, overlay_options: dict):
        """
        Overlays the video onto a background, creating a mask for rounded corners if needed.
        """
        opts = {**{'corner_radius': 0}, **overlay_options}
        logger.info(f"Applying background with options: {opts}")

        mask_path = None
        temp_mask_file = None
        try:
            if opts['corner_radius'] > 0:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_mask_file:
                    mask_path = temp_mask_file.name
                
                mask_created = await self._create_rounded_mask_image_async(
                    opts['width'], opts['height'], opts['corner_radius'], mask_path
                )
                if not mask_created:
                    raise ValueError("Mask image creation failed.")

            return await self.ffmpeg_processor.apply_background(
                input_video_path, background_image_path, output_video_path, opts, mask_path
            )

        except Exception as e:
            logger.error(f"An exception occurred while applying frame: {e}")
            return False
        finally:
            if mask_path and os.path.exists(mask_path):
                os.remove(mask_path)
                logger.info(f"Cleaned up temporary mask file: {mask_path}")

    async def process_video_segments_async(self, input_video: str, optimized_segments: List[OptimizedSegment]) -> str:
        """
        Async processing of video segments in a two-stage process.
        Stage 1: Generate all audio concurrently.
        Stage 2: Process all video segments concurrently.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing video with {len(optimized_segments)} segments in {temp_dir}")

            # --- PRE-FLIGHT CHECK ---
            video_duration = await self.ffmpeg_processor.get_media_duration(input_video)
            logger.info(f"Input video duration check: {video_duration:.2f}s")
            if video_duration == 0.0:
                if not os.path.exists(input_video):
                    raise ValueError(f"Input video file does not exist: {input_video}")
                raise ValueError(f"Could not read video duration. File may be corrupted: {input_video}")

            # --- STAGE 1: GENERATE ALL AUDIO FILES CONCURRENTLY ---
            logger.info("--- Stage 1: Generating all audio files ---")
            audio_tasks = []
            for i, opt_seg in enumerate(optimized_segments):
                segment_temp_dir = os.path.join(temp_dir, f"segment_{i}")
                os.makedirs(segment_temp_dir, exist_ok=True)
                audio_tasks.append(
                    self.generate_tts_audio_async(opt_seg.optimized_text, segment_temp_dir)
                )
            
            audio_generation_results = await asyncio.gather(*audio_tasks, return_exceptions=True)

            # Collect successful audio results to pass to the next stage
            successful_audio_data = []
            for i, result in enumerate(audio_generation_results):
                if isinstance(result, Exception) or not result or result[0] is None or result[1] <= 0:
                    logger.warning(f"Skipping segment {i} due to audio generation failure.")
                    continue
                audio_path, audio_duration = result
                successful_audio_data.append({
                    "index": i,
                    "opt_seg": optimized_segments[i],
                    "audio_path": audio_path,
                    "audio_duration": audio_duration
                })

            if not successful_audio_data:
                raise ValueError("All audio generation tasks failed. Cannot proceed.")
            logger.info(f"--- Stage 1 Complete: Successfully generated {len(successful_audio_data)} audio files ---")


            # --- STAGE 2: PROCESS ALL VIDEO SEGMENTS CONCURRENTLY ---
            logger.info("--- Stage 2: Processing all video segments using generated audio data ---")
            semaphore = asyncio.Semaphore(2)  # Limit concurrent FFmpeg processes

            async def process_single_video_segment(segment_data: dict):
                async with semaphore:
                    i = segment_data["index"]
                    opt_seg = segment_data["opt_seg"]
                    actual_audio_duration = segment_data["audio_duration"]
                    segment_temp_dir = os.path.dirname(segment_data["audio_path"])

                    segment_video_path = os.path.join(segment_temp_dir, f"video_{i:04d}.mp4")
                    start_time = opt_seg.original.start_seconds
                    original_video_seg_duration = opt_seg.original.duration
                    speed_multiplier = original_video_seg_duration / actual_audio_duration

                    try:
                        success = await self.ffmpeg_processor.extract_and_retime_segment(
                            input_video, segment_video_path, start_time,
                            original_video_seg_duration, speed_multiplier
                        )
                        if success:
                            logger.info(f"Segment {i+1}: Speed {speed_multiplier:.2f}x, Duration: {actual_audio_duration:.2f}s")
                            return segment_video_path, segment_data["audio_path"]
                        else:
                            return None
                    except Exception as e:
                        logger.error(f"Exception processing video segment {i}: {e}")
                        return None

            video_tasks = [process_single_video_segment(data) for data in successful_audio_data]
            video_processing_results = await asyncio.gather(*video_tasks, return_exceptions=True)
            logger.info("--- Stage 2 Complete: Video segment processing finished ---")
            
            # --- STAGE 3: COLLECT RESULTS AND COMBINE ---
            video_segments = []
            audio_segments = []
            for result in video_processing_results:
                if result and not isinstance(result, Exception):
                    video_path, audio_path = result
                    video_segments.append(video_path)
                    audio_segments.append(audio_path)
            
            if not video_segments:
                raise ValueError("No valid video segments were created after processing stage.")

            # Combine all processed segments into a temporary video file
            internal_output_video = os.path.join(temp_dir, "final_output.mp4")
            await self._robust_combine_segments_async(video_segments, audio_segments, internal_output_video, temp_dir)
            
            # Move the final video to a persistent path that won't be auto-deleted
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
                persistent_output_path = dest_f.name
            shutil.move(internal_output_video, persistent_output_path)
            
            logger.info(f"Video processing complete. Final output at: {persistent_output_path}")
            return persistent_output_path
            
    async def _robust_combine_segments_async(self, video_segments: List[str], audio_segments: List[str], 
                                             output_video: str, temp_dir: str):
        """Async combination of video and audio segments using robust concatenation method"""
        if len(video_segments) != len(audio_segments):
            raise ValueError(f"Mismatch: {len(video_segments)} video vs {len(audio_segments)} audio segments")
        
        logger.info(f"Combining {len(video_segments)} segments with audio-priority timing...")
        
        combined_segments_for_concat = []
        
        # Stage 1: Create combined segments with audio
        async def combine_single_segment(i, video_seg_path, audio_seg_path):
            final_segment_path = os.path.join(temp_dir, f"combined_{i:04d}.mp4")
            audio_dur = await self.ffmpeg_processor.get_media_duration(audio_seg_path)
            if audio_dur == 0.0:
                logger.warning(f"Skipping segment {i} due to zero duration audio.")
                return None
            
            success = await self.ffmpeg_processor.combine_video_audio_segment(
                video_seg_path, audio_seg_path, final_segment_path, audio_dur
            )
            return final_segment_path if success else None

        combine_tasks = [
            combine_single_segment(i, v, a) for i, (v, a) in enumerate(zip(video_segments, audio_segments))
        ]
        combine_results = await asyncio.gather(*combine_tasks, return_exceptions=True)
        
        for result in combine_results:
            if result and not isinstance(result, Exception):
                combined_segments_for_concat.append(Path(result))
        
        if not combined_segments_for_concat:
            raise ValueError("No segments were successfully combined.")
        
        # Stage 2: Use robust concatenation method
        temp_ts_dir = Path(temp_dir) / "temp_ts_files"
        temp_ts_dir.mkdir(exist_ok=True)
        
        # Sort the combined segments naturally
        combined_segments_for_concat.sort(key=self._natural_sort_key)
        
        # Sanitize clips to TS format
        async def sanitize_single_clip(i, clip_path):
            output_ts_path = temp_ts_dir / f"clean_{i:04d}.ts"
            success = await self.ffmpeg_processor.sanitize_clip_to_ts(clip_path, output_ts_path)
            return output_ts_path if success else None

        sanitize_tasks = [sanitize_single_clip(i, p) for i, p in enumerate(combined_segments_for_concat)]
        sanitize_results = await asyncio.gather(*sanitize_tasks)
        clean_ts_files = [res for res in sanitize_results if res]
        
        if not clean_ts_files or len(clean_ts_files) != len(combined_segments_for_concat):
            logger.error("Aborting due to failure in sanitization stage.")
            raise ValueError("Failed to sanitize all video segments")
        
        # Concatenate TS files
        success = await self.ffmpeg_processor.concatenate_ts_files(clean_ts_files, output_video, temp_dir)
        
        if not success:
            raise ValueError("Failed to concatenate segments")
        
        logger.info("Successfully combined all segments using robust method.")
    
    def _natural_sort_key(self, filename: Path) -> int:
        """Natural sorting key for combined video files"""
        match = re.search(r'combined_(\d+)\.mp4', filename.name)
        if match:
            return int(match.group(1))
        return 0
    
    def get_cost_summary(self) -> Dict:
        """Return cost and usage summary"""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": self.total_cost
        }

    async def process_srt_content_to_video_async(
        self,
        srt_content: str,
        input_video: str,
        background_image_path: str = None,
        overlay_options: Dict = None,
    ) -> tuple:
        """
        Main async method to process SRT content and create optimized video.
        Optionally applies a background image/frame to the final video.
        """
        logger.info("Starting SRT content processing for video optimization")
    
        try:
            segments = self.parse_srt_content(srt_content)
            if not segments:
                raise ValueError("No valid segments found in SRT content")
    
            optimized_segments = await self.optimize_all_segments_async(segments)
    
            intermediate_video_path = await self.process_video_segments_async(
                input_video, optimized_segments
            )
    
            final_video_path = intermediate_video_path
            print(f"this si the fnal video path{final_video_path}")
    
            if background_image_path:
                print(f"this is the backgorundimage path{background_image_path}")
                if not os.path.exists(background_image_path):
                    logger.error(
                        f"Background image not found at {background_image_path}. Skipping."
                    )
                else:
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ) as dest_f:
                        composited_video_path = dest_f.name
    
                    # Use provided overlay options, or an empty dict for defaults
                    current_overlay_options = overlay_options or {}
    
                    success = await self._apply_background_async(
                        intermediate_video_path,
                        background_image_path,
                        composited_video_path,
                        current_overlay_options,
                    )
    
                    if success:
                        final_video_path = composited_video_path
                        os.remove(intermediate_video_path)
                    else:
                        logger.warning(
                            "Could not apply frame. Returning the original video."
                        )
                        os.remove(composited_video_path)
            
            cost_summary = self.get_cost_summary()
    
            logger.info("SRT processing and video optimization complete")
            return final_video_path, cost_summary
    
        except Exception as e:
            logger.error(f"Error in SRT processing: {e}")
            raise

    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)