import os
import asyncio
import logging
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from app.config import get_settings

logger = logging.getLogger(__name__)

class ProgressPercentage(object):
    """A callback class to display the upload progress."""
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._last_logged_progress = -1

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            # Log progress at intervals (e.g., every 5%) to avoid spamming logs
            if int(percentage) > self._last_logged_progress and int(percentage) % 5 == 0:
                self._last_logged_progress = int(percentage)
                logger.info(f"Upload progress for {os.path.basename(self._filename)}: {percentage:.2f}%")

class CloudflareR2Uploader:
    """Async Cloudflare R2 uploader service, mimicking GoogleDriveUploader structure."""
    
    def __init__(self):
        settings = get_settings()
        self.bucket_name = settings.r2_bucket_name
        self.public_domain = settings.r2_public_domain
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.s3_client = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{settings.cloudflare_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.cloudflare_access_key_id,
            aws_secret_access_key=settings.cloudflare_secret_access_key,
            config=Config(signature_version='s3v4')
        )
        logger.info(f"Cloudflare R2 uploader initialized. Bucket: {self.bucket_name}")

    def _upload_file_sync(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Synchronous file upload to R2, returning a Google Drive-like format."""
        try:
            if not os.path.exists(file_path):
                return {'success': False, 'error': f"File not found: {file_path}"}
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Starting R2 upload: {filename} ({file_size / (1024*1024):.2f} MB)")
            
            # Use filename as the object key in R2
            object_name = filename
            
            progress_callback = ProgressPercentage(file_path)

            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                object_name,
                Callback=progress_callback
            )

            # Generate URLs
            public_url = f"https://{self.public_domain}/{object_name}"
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=3600  # Expires in 1 hour
            )
            
            logger.info(f"R2 Upload completed: {filename} (Key: {object_name})")
            
            # Mimic the Google Drive Uploader's successful return format
            return {
                'success': True,
                'file_id': object_name,
                'file_name': filename,
                'web_view_link': public_url,
                'download_link': download_url,
                'size': file_size
            }
            
        except (BotoCoreError, ClientError) as error:
            error_message = f"R2 API error: {error}"
            logger.error(error_message)
            return {'success': False, 'error': error_message}
        except Exception as error:
            error_message = f"R2 Upload failed: {error}"
            logger.error(error_message)
            return {'success': False, 'error': error_message}

    async def upload_video_async(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Async wrapper for video upload to R2."""
        if not os.path.exists(file_path):
            return {'success': False, 'error': f"File not found: {file_path}"}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._upload_file_sync,
            file_path,
            filename
        )

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)