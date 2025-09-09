import re
import json
import os
import logging
import time
import asyncio
import tempfile
import shutil
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import openai
import subprocess
from pydantic import BaseModel, Field
import aiohttp
import requests
from requests.exceptions import RequestException
from murf import Murf
from gtts import gTTS
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq
load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI Model Pricing (USD per 1 million tokens)
MODEL_PRICING = {
    "gpt-4.1-mini": {"prompt_per_million": 0.40, "completion_per_million": 1.6},
}

# --- Pydantic Models for Structured Data ---

class PolishedSentenceGroup(BaseModel):
    """Defines the structure for a single polished sentence group from the LLM."""
    segment_indices: List[int] = Field(..., description="A list of original segment indices that form this sentence.")
    polished_sentence: str = Field(..., description="The final, merged, and polished sentence text, rewritten for clarity and flow.")

class SentenceGroupingResponse(BaseModel):
    """The expected JSON response from the LLM for sentence grouping."""
    sentences: List[PolishedSentenceGroup] = Field(..., description="A list of polished sentence groups.")

# --- Dataclasses for Internal State ---

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
    """Represents an initially optimized text segment"""
    original: SRTSegment
    optimized_text: str

@dataclass
class SentenceGroup:
    """Represents a group of original segments that form a complete sentence."""
    index: int
    grouped_text: str
    original_segments: List[SRTSegment]
    start_seconds: float
    end_seconds: float
    original_duration: float

# --- Core Classes ---

class FFmpegProcessor:
    """Handles all FFmpeg and ffprobe subprocess executions asynchronously."""
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

    async def extract_and_retime_segment(
        self, input_video: str, output_path: str, start_time: float,
        original_duration: float, speed_multiplier: float
    ) -> bool:
        """Extracts and retimes a single video segment."""
        end_time = start_time + original_duration
        safe_multiplier = max(0.25, min(4.0, speed_multiplier))
        if abs(safe_multiplier - speed_multiplier) > 0.01:
            logger.warning(f"Clamped speed multiplier from {speed_multiplier:.2f} to {safe_multiplier:.2f}")

        extract_cmd = [
            'ffmpeg', '-y', 
            '-ss', f'{start_time:.3f}', 
            '-to', f'{end_time:.3f}',
            '-i', input_video,
            '-filter_complex', f'[0:v]setpts={1/safe_multiplier:.4f}*PTS[v]',
            '-map', '[v]',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-an', output_path
        ]
        
        def extract_video_sync():
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=300)
            return result.returncode, result.stderr

        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(self.executor, extract_video_sync)
        
        if returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logger.error(f"FFmpeg failed for segment extraction (code {returncode}): {stderr}")
            logger.error(f"Command: {' '.join(extract_cmd)}")
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

    async def concatenate_files(self, file_list_path: str, output_video: str) -> bool:
        """Concatenates a list of video files using the concat demuxer."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            output_video
        ]
        
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


class VideoSummarizer:
    """Async video summarizer that handles multiple concurrent requests"""
    
    def __init__(self, azure_openai_key: str, azure_openai_endpoint: str, 
                 azure_api_version: str, azure_deployment_name: str, 
                 murf_api_key: str = None, tts_engine: str = 'gtts', 
                 target_wpm: float = 160.0,
                 vibe_voice_config: Optional[Dict] = None,
                 groq_playai_config: Optional[Dict] = None):
        
        self.openai_client = openai.AzureOpenAI(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_api_version
        )
        self.azure_deployment_name = azure_deployment_name
        
        self.tts_engine = tts_engine.lower()
        
        # Initialize TTS engines based on configuration
        if self.tts_engine == 'murf':
            if not murf_api_key:
                raise ValueError("Murf API key is required when using the 'murf' TTS engine.")
            self.murf_client = Murf(api_key=murf_api_key)
        
        elif self.tts_engine == 'vibe_voice':
            if not vibe_voice_config:
                raise ValueError("Vibe Voice configuration is required when using the 'vibe_voice' TTS engine.")
            self.vibe_voice_config = vibe_voice_config
            required_keys = ['api_url', 'model_path', 'speaker_name']
            missing_keys = [key for key in required_keys if key not in vibe_voice_config]
            if missing_keys:
                raise ValueError(f"Missing Vibe Voice configuration keys: {missing_keys}")
        
        elif self.tts_engine == 'groq_playai':
            if not groq_playai_config:
                raise ValueError("Groq PlayAI configuration is required when using the 'groq_playai' TTS engine.")
            self.groq_playai_config = groq_playai_config
            required_keys = ['api_key', 'voice']
            missing_keys = [key for key in required_keys if key not in groq_playai_config]
            if missing_keys:
                raise ValueError(f"Missing Groq PlayAI configuration keys: {missing_keys}")
            self.groq_client = Groq(api_key=groq_playai_config['api_key'])
            
            # Rate limiting for Groq PlayAI (10 requests per minute max)
            self.groq_request_times = []
            self.groq_rate_limit = 10  # requests per minute
            self.groq_rate_window = 60  # seconds
        
        self.target_wpm = target_wpm
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.ffmpeg_processor = FFmpegProcessor(self.executor)

    def _update_usage_and_cost(self, usage, model: str = "gpt-4.1-mini"):
        """Calculates cost for a single API call and updates totals."""
        if not usage:
            logger.warning("No usage data found in OpenAI response.")
            return
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        model_rates = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4.1-mini"])
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
            if not block.strip(): continue
            lines = block.strip().split('\n')
            if len(lines) < 3: continue
            try:
                index = int(lines[0])
                time_line = lines[1]
                text = ' '.join(lines[2:]).strip()
                time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
                if not time_match: continue
                start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, time_match.groups())
                start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                duration = end_seconds - start_seconds
                word_count = len(text.split())
                words_per_second = word_count / duration if duration > 0 else 0
                segments.append(SRTSegment(index=index, start_time=f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}", end_time=f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}", start_seconds=start_seconds, end_seconds=end_seconds, duration=duration, text=text, word_count=word_count, words_per_second=words_per_second))
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed segment: {e}")
                continue
        logger.info(f"Parsed {len(segments)} segments")
        return segments

    async def optimize_all_segments_async(self, segments: List[SRTSegment]) -> List[OptimizedSegment]:
        """Performs the initial, segment-by-segment text cleanup."""
        logger.info("Starting initial optimization of all segments...")
        optimized_segments = [
            OptimizedSegment(
                original=segment, 
                optimized_text=re.sub(r'\s+', ' ', segment.text).strip()
            ) for segment in segments
        ]
        logger.info("Initial optimization complete.")
        return optimized_segments

    async def group_segments_into_sentences_async(self, optimized_segments: List[OptimizedSegment]) -> List[SentenceGroup]:
        """Uses an LLM to group segments into sentences AND return the polished, merged text directly."""
        logger.info("Grouping segments into polished sentences...")
        
        formatted_texts = "\n".join([f"{i}: {seg.optimized_text}" for i, seg in enumerate(optimized_segments)])
        
        json_schema = SentenceGroupingResponse.model_json_schema()

        grouping_prompt = f"""
        You are an expert video script editor. I have a list of transcribed text segments from a video.
        Your task is to group these segments into complete, natural sentences. Then, for each group, merge and polish the text into a single, flowing sentence.

        Analyze the list of numbered text segments below. Then, respond ONLY with a single JSON object that provides:
        1. The indices of the original segments that form the sentence.
        2. The final `polished_sentence` after merging the text and rewriting it for maximum clarity and natural flow.

        The JSON schema you MUST follow is:
        {json.dumps(json_schema, indent=2)}

        For example, if segments 0 and 1 form one sentence, your output should be:
        {{
            "sentences": [
                {{ 
                    "segment_indices": [0, 1],
                    "polished_sentence": "This is the beautifully merged and rewritten text for segments 0 and 1."
                }}
            ]
        }}

        Here are the text segments:
        ---
        {formatted_texts}
        ---
        """
        
        try:
            loop = asyncio.get_event_loop()
            def make_openai_call():
                return self.openai_client.chat.completions.create(
                    model=self.azure_deployment_name,
                    messages=[{"role": "user", "content": grouping_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
            
            response = await loop.run_in_executor(self.executor, make_openai_call)
            self._update_usage_and_cost(response.usage)
            
            response_json = json.loads(response.choices[0].message.content)
            grouping_data = SentenceGroupingResponse(**response_json)
            
            sentence_groups = []
            for i, group_info in enumerate(grouping_data.sentences):
                indices = group_info.segment_indices
                if not indices: continue
                
                original_segs_in_group = [optimized_segments[j].original for j in indices]
                
                start_seconds = original_segs_in_group[0].start_seconds
                end_seconds = original_segs_in_group[-1].end_seconds
                original_duration = end_seconds - start_seconds

                sentence_groups.append(SentenceGroup(
                    index=i,
                    grouped_text=group_info.polished_sentence,
                    original_segments=original_segs_in_group,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    original_duration=original_duration
                ))
            
            logger.info(f"Successfully grouped and polished {len(optimized_segments)} segments into {len(sentence_groups)} sentences.")
            return sentence_groups

        except Exception as e:
            logger.error(f"Failed to group segments into sentences: {e}. Aborting.")
            raise

    def _generate_vibe_voice_tts(self, text: str, output_path: str) -> bool:
        """Generate TTS using Vibe Voice API."""
        try:
            api_url = self.vibe_voice_config['api_url']
            model_path = self.vibe_voice_config['model_path']
            speaker_name = self.vibe_voice_config['speaker_name']
            
            formatted_text = f"Speaker 1: {text}"
            temp_id = str(uuid.uuid4())
            temp_txt_file = f"temp_{temp_id}.txt"
            
            try:
                with open(temp_txt_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
                
                with open(temp_txt_file, 'rb') as txt_file:
                    files = {
                        'txt_file': ('text.txt', txt_file, 'text/plain')
                    }
                    
                    data = {
                        'model_path': model_path,
                        'speaker_names': [speaker_name]
                    }
                    
                    response = requests.post(api_url, files=files, data=data, timeout=120)
                    
                    if response.status_code == 200:
                        with open(output_path, 'wb') as audio_file:
                            audio_file.write(response.content)
                        return True
                    else:
                        logger.error(f"Vibe Voice API request failed with status code: {response.status_code}")
                        return False
            
            finally:
                if os.path.exists(temp_txt_file):
                    os.remove(temp_txt_file)
            
        except Exception as e:
            logger.error(f"Vibe Voice TTS generation failed: {e}")
            return False

    def _wait_for_groq_rate_limit(self):
        """Ensure we don't exceed Groq PlayAI rate limits (10 requests per minute)."""
        current_time = time.time()
        
        # Remove requests older than the rate window
        self.groq_request_times = [
            req_time for req_time in self.groq_request_times 
            if current_time - req_time < self.groq_rate_window
        ]
        
        # If we're at the limit, wait until we can make another request
        if len(self.groq_request_times) >= self.groq_rate_limit:
            oldest_request = min(self.groq_request_times)
            wait_time = self.groq_rate_window - (current_time - oldest_request) + 1  # +1 for safety buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds before next Groq PlayAI request...")
                time.sleep(wait_time)
        
        # Record this request
        self.groq_request_times.append(current_time)

    def _generate_groq_playai_tts(self, text: str, output_path: str) -> bool:
        """Generate TTS using Groq PlayAI with rate limiting."""
        try:
            # Apply rate limiting
            self._wait_for_groq_rate_limit()
            
            voice = self.groq_playai_config['voice']
            
            logger.info(f"Making Groq PlayAI TTS request (Request #{len(self.groq_request_times)} in current window)")
            
            response = self.groq_client.audio.speech.create(
                model="playai-tts",
                voice=voice,
                input=text,
                response_format="wav"
            )
            
            response.write_to_file(output_path)
            logger.info(f"Successfully generated Groq PlayAI TTS audio")
            return True
            
        except Exception as e:
            logger.error(f"Groq PlayAI TTS generation failed: {e}")
            return False

    async def generate_tts_audio_async(self, text: str, temp_dir: str, voice_id: str = 'en-IN-arohi', 
                                         retries: int = 3, delay: int = 5) -> tuple:
        """
        Async generation of TTS audio to a temporary file inside a given directory.
        Returns: (final_file_path, duration)
        """
        if not text.strip():
            logger.warning("Empty text provided for TTS, skipping.")
            return None, 0.0
    
        audio_filename = "audio.mp3" if self.tts_engine in ['murf', 'gtts'] else "audio.wav"
        output_path = os.path.join(temp_dir, audio_filename)
        
        if self.tts_engine == 'murf':
            for attempt in range(retries):
                try:
                    logger.info(f"Generating Murf TTS for: '{text[:40]}...'")
                    
                    loop = asyncio.get_event_loop()
                    
                    def generate_murf_tts():
                        res = self.murf_client.text_to_speech.generate(
                            text=text,
                            voice_id=voice_id,
                            format='mp3'
                        )
                        return res.audio_file, res.audio_length_in_seconds
                    
                    audio_url, audio_duration = await loop.run_in_executor(self.executor, generate_murf_tts)
                    
                    if not audio_url:
                        logger.error("Murf API did not return an audio URL.")
                        return None, 0.0
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(audio_url) as response:
                            response.raise_for_status()
                            with open(output_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                    
                    logger.info(f"Successfully saved Murf TTS audio to {output_path}")
                    final_duration = await self.ffmpeg_processor.get_media_duration(output_path)
                    return output_path, final_duration
                    
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

        elif self.tts_engine == 'vibe_voice':
            try:
                logger.info(f"Generating Vibe Voice TTS for: '{text[:40]}...'")
                
                loop = asyncio.get_event_loop()
                
                generation_success = await loop.run_in_executor(
                    self.executor, 
                    self._generate_vibe_voice_tts, 
                    text, 
                    output_path
                )
                
                if generation_success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    duration = await self.ffmpeg_processor.get_media_duration(output_path)
                    logger.info(f"Successfully saved Vibe Voice TTS audio to {output_path} (Duration: {duration:.2f}s)")
                    return output_path, duration
                else:
                    logger.error(f"Vibe Voice TTS failed to create a valid audio file")
                    return None, 0.0
                    
            except Exception as e:
                logger.error(f"Vibe Voice TTS generation failed: {e}")
                return None, 0.0

        elif self.tts_engine == 'groq_playai':
            try:
                logger.info(f"Generating Groq PlayAI TTS for: '{text[:40]}...'")
                
                loop = asyncio.get_event_loop()
                
                generation_success = await loop.run_in_executor(
                    self.executor, 
                    self._generate_groq_playai_tts, 
                    text, 
                    output_path
                )
                
                if generation_success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    duration = await self.ffmpeg_processor.get_media_duration(output_path)
                    logger.info(f"Successfully saved Groq PlayAI TTS audio to {output_path} (Duration: {duration:.2f}s)")
                    return output_path, duration
                else:
                    logger.error(f"Groq PlayAI TTS failed to create a valid audio file")
                    return None, 0.0
                    
            except Exception as e:
                logger.error(f"Groq PlayAI TTS generation failed: {e}")
                return None, 0.0
        
        else:
            logger.error(f"TTS generation skipped: unknown engine '{self.tts_engine}'")
            return None, 0.0

    def _create_rounded_mask_image(self, width: int, height: int, radius: int, output_path: str):
        """Creates a black PNG with a white rounded rectangle to use as a video mask."""
        try:
            mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask)
            
            draw.rounded_rectangle(
                (0, 0, width, height),
                fill=(255, 255, 255, 255),
                radius=radius
            )
            mask.save(output_path, 'PNG')
            logger.info(f"Successfully created temporary mask at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create mask image: {e}")
            return False

    async def _apply_background_async(self, input_video_path: str, background_image_path: str,
                                      output_video_path: str, overlay_options: dict):
        """
        Overlays the video onto a background using a robust PNG mask for rounded corners.
        This version adds a crop filter to ensure final dimensions are even, fixing encoder errors.
        """
        defaults = {
            'width': 1280, 'height': 720, 'x': '(main_w-overlay_w)/2',
            'y': '(main_h-overlay_h)/2', 'corner_radius': 0
        }
        opts = {**defaults, **overlay_options}
        logger.info(f"Applying background with options: {opts}")

        mask_path = None
        temp_mask_file = None
        try:
            filter_chain_prefix = "[0:v]crop=floor(iw/2)*2:floor(ih/2)*2[bg];"

            if opts['corner_radius'] > 0:
                temp_mask_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                mask_path = temp_mask_file.name
                temp_mask_file.close()

                if not self._create_rounded_mask_image(opts['width'], opts['height'], opts['corner_radius'], mask_path):
                    raise ValueError("Mask image creation failed.")

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
                    '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
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
                    '-c:a', 'copy', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
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
        except Exception as e:
            logger.error(f"An exception occurred while applying frame: {e}")
            return False
        finally:
            if mask_path and os.path.exists(mask_path):
                os.remove(mask_path)
                logger.info(f"Cleaned up temporary mask file: {mask_path}")

    async def process_sentence_groups_async(self, input_video: str, sentence_groups: List[SentenceGroup]) -> str:
        """Processes video based on sentence groups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing {len(sentence_groups)} sentence groups in {temp_dir}")
            
            semaphore = asyncio.Semaphore(4)
            final_clips_for_concat = [None] * len(sentence_groups)
    
            async def process_single_group(group: SentenceGroup):
                async with semaphore:
                    i = group.index
                    segment_temp_dir = os.path.join(temp_dir, f"group_{i}")
                    os.makedirs(segment_temp_dir, exist_ok=True)
                    
                    generated_audio_path, new_audio_duration = await self.generate_tts_audio_async(
                        group.grouped_text, segment_temp_dir
                    )
                    
                    if not generated_audio_path or new_audio_duration <= 0.01:
                        logger.warning(f"Skipping group {i} due to audio generation failure.")
                        return
    
                    speed_multiplier = group.original_duration / new_audio_duration
                    
                    retimed_video_path = os.path.join(segment_temp_dir, f"retimed_video_{i}.mp4")
                    extract_success = await self.ffmpeg_processor.extract_and_retime_segment(
                        input_video, retimed_video_path, group.start_seconds,
                        group.original_duration, speed_multiplier
                    )
                    
                    if not extract_success:
                        logger.warning(f"Skipping group {i} due to video re-timing failure.")
                        return
    
                    final_clip_path = os.path.join(temp_dir, f"final_clip_{i:04d}.mp4")
                    combine_success = await self.ffmpeg_processor.combine_video_audio_segment(
                        retimed_video_path, generated_audio_path, final_clip_path, new_audio_duration
                    )
                    
                    if combine_success:
                        logger.info(f"Group {i+1}/{len(sentence_groups)} processed. Speed: {speed_multiplier:.2f}x")
                        final_clips_for_concat[i] = final_clip_path
                    else:
                        logger.warning(f"Skipping group {i} due to final combination failure.")
            
            tasks = [process_single_group(group) for group in sentence_groups]
            await asyncio.gather(*tasks)
    
            valid_final_clips = [clip for clip in final_clips_for_concat if clip]
            if not valid_final_clips:
                raise ValueError("No video clips were successfully processed.")
    
            concat_list_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_path, 'w') as f:
                for clip_path in valid_final_clips:
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")
    
            internal_output_video = os.path.join(temp_dir, "final_video.mp4")
            concat_success = await self.ffmpeg_processor.concatenate_files(concat_list_path, internal_output_video)
    
            if not concat_success:
                raise RuntimeError("Final video concatenation failed.")
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
                persistent_output_path = dest_f.name
            shutil.move(internal_output_video, persistent_output_path)
            
            logger.info(f"Video processing complete. Final output at: {persistent_output_path}")
            return persistent_output_path

    async def process_srt_content_to_video_async(self, srt_content: str, input_video: str) -> tuple:
        """Main orchestrator method using the new sentence grouping approach."""
        logger.info("Starting SRT processing with sentence-grouping workflow.")
    
        try:
            segments = self.parse_srt_content(srt_content)
            if not segments: raise ValueError("No valid segments found in SRT content")
    
            optimized_segments = await self.optimize_all_segments_async(segments)
    
            sentence_groups = await self.group_segments_into_sentences_async(optimized_segments)
    
            final_video_path = await self.process_sentence_groups_async(input_video, sentence_groups)
            
            cost_summary = self.get_cost_summary()
    
            logger.info("SRT processing and video optimization complete!")
            return final_video_path, cost_summary
    
        except Exception as e:
            logger.error(f"Error in main processing pipeline: {e}")
            raise

    def get_cost_summary(self) -> Dict:
        """Return cost and usage summary"""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": self.total_cost
        }

    def __del__(self):
        if hasattr(self, 'executor'): self.executor.shutdown(wait=False)

# # EXAMPLE USAGE:
# async def main():

#     AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
#     AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
#     AZURE_API_VERSION = "2025-01-01-preview"
#     AZURE_DEPLOYMENT_NAME = "gpt-4.1-mini"  # Your Azure deployment name
    
#     INPUT_VIDEO_PATH = r"C:\Users\amand\Videos\2025-03-19 01-56-12.mkv"
#     SRT_FILE_PATH = r"C:\Users\amand\Videos\2025-03-19 01-56-12.srt"
    
#     if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT]):
#         print("Error: Azure OpenAI credentials are not set in environment variables.")
#         return
#     if not os.path.exists(INPUT_VIDEO_PATH):
#         print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
#         return
#     if not os.path.exists(SRT_FILE_PATH):
#         print(f"Error: SRT file not found at {SRT_FILE_PATH}")
#         return
        
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     # Test different TTS engines - uncomment the one you want to test
    
#     # Option 1: Test with Murf TTS
#     # summarizer = VideoSummarizer(
#     #     azure_openai_key=AZURE_OPENAI_KEY,
#     #     azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
#     #     azure_api_version=AZURE_API_VERSION,
#     #     azure_deployment_name=AZURE_DEPLOYMENT_NAME,
#     #     tts_engine='murf',
#     #     murf_api_key="ap2_2457c316-893a-4599-88cc-3a31aecd29c7"
#     # )
    
#     # Option 2: Test with gTTS
#     summarizer = VideoSummarizer(
#         azure_openai_key=AZURE_OPENAI_KEY,
#         azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
#         azure_api_version=AZURE_API_VERSION,
#         azure_deployment_name=AZURE_DEPLOYMENT_NAME,
#         tts_engine='gtts'
#     )
    
#     # Option 3: Test with Vibe Voice TTS
#     # vibe_voice_config = {
#     #     'api_url': 'https://fyzy94d0jqdy.share.zrok.io/generate-audio/',
#     #     'model_path': 'VibeVoice-1.5B',
#     #     'speaker_name': 'Alice_woman'  # Available speakers: Alice_woman, Bob_man, etc.
#     # }
#     # summarizer = VideoSummarizer(
#     #     azure_openai_key=AZURE_OPENAI_KEY,
#     #     azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
#     #     azure_api_version=AZURE_API_VERSION,
#     #     azure_deployment_name=AZURE_DEPLOYMENT_NAME,
#     #     tts_engine='vibe_voice',
#     #     vibe_voice_config=vibe_voice_config
#     # )
    
#     # # Option 4: Test with Groq PlayAI TTS
#     # groq_playai_config = {
#     #     'api_key': os.environ.get("GROQ_API_KEY"),  # Make sure to set this environment variable
#     #     'voice': 'Fritz-PlayAI' 
#     # }
    
#     # if not groq_playai_config['api_key']:
#     #     print("Error: GROQ_API_KEY environment variable is not set.")
#     #     return
    
#     # summarizer = VideoSummarizer(
#     #     azure_openai_key=AZURE_OPENAI_KEY,
#     #     azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
#     #     azure_api_version=AZURE_API_VERSION,
#     #     azure_deployment_name=AZURE_DEPLOYMENT_NAME,
#     #     tts_engine='groq_playai',
#     #     groq_playai_config=groq_playai_config
#     # )

#     with open(SRT_FILE_PATH, 'r', encoding='utf-8') as f:
#         srt_content = f.read()

#     try:
#         print(f"Starting video processing with {summarizer.tts_engine.upper()} TTS engine...")
#         final_video, cost = await summarizer.process_srt_content_to_video_async(srt_content, INPUT_VIDEO_PATH)
#         print("\n--- Process Complete ---")
#         print(f"Final video created at: {final_video}")
#         print(f"TTS Engine Used: {summarizer.tts_engine.upper()}")
#         print(f"Estimated OpenAI Cost: ${cost['estimated_cost_usd']:.6f}")
#         print(f"Total Tokens Used: {cost['total_tokens']}")
#     except Exception as e:
#         print(f"\n--- An Error Occurred ---")
#         print(f"Error: {e}")
        
# if __name__ == "__main__":
#     asyncio.run(main())