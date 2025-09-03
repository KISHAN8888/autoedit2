# app/services/google_drive_uploader.py
import os
import asyncio
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class GoogleDriveUploader:
    """Async Google Drive uploader service"""
    
    def __init__(self, service_account_path: str, folder_id: str = None):
        self.service_account_path = service_account_path
        self.folder_id = folder_id
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Scopes required for Google Drive API
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        
        # Validate service account file exists
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Service account file not found: {service_account_path}")
        
        logger.info(f"Google Drive uploader initialized. Folder ID: {folder_id or 'Root'}")
        
    def _get_drive_service(self):
        """Create and return Google Drive service instance"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path, scopes=self.scopes
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to create Google Drive service: {e}")
            raise
    
    def _upload_file_sync(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Synchronous file upload to Google Drive"""
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"File not found: {file_path}"
                }
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Starting upload: {filename} ({file_size} bytes)")
            
            service = self._get_drive_service()
            
            # File metadata
            file_metadata = {'name': filename}
            if self.folder_id:
                file_metadata['parents'] = [self.folder_id]
            
            # Upload file with resumable upload for large files
            media = MediaFileUpload(
                file_path, 
                resumable=True,
                chunksize=1024*1024  # 1MB chunks
            )
            
            request = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink,webContentLink,size'
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")
            
            logger.info(f"Upload completed: {response['name']} (ID: {response['id']})")
            
            return {
                'success': True,
                'file_id': response['id'],
                'file_name': response['name'],
                'web_view_link': response['webViewLink'],
                'download_link': response['webContentLink'],
                'size': response.get('size', 0)
            }
            
        except HttpError as error:
            error_message = f"Google Drive API error: {error}"
            logger.error(error_message)
            return {
                'success': False,
                'error': error_message
            }
        except Exception as error:
            error_message = f"Upload failed: {error}"
            logger.error(error_message)
            return {
                'success': False,
                'error': error_message
            }
    
    async def upload_video_async(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Async wrapper for video upload to Google Drive"""
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._upload_file_sync,
            file_path,
            filename
        )
    
    def _delete_file_sync(self, file_id: str) -> Dict[str, Any]:
        """Synchronously delete a file from Google Drive"""
        try:
            service = self._get_drive_service()
            service.files().delete(fileId=file_id).execute()
            
            logger.info(f"File deleted: {file_id}")
            return {'success': True}
            
        except HttpError as error:
            error_message = f"Failed to delete file {file_id}: {error}"
            logger.error(error_message)
            return {
                'success': False,
                'error': error_message
            }
    
    async def delete_file_async(self, file_id: str) -> Dict[str, Any]:
        """Async wrapper for file deletion"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._delete_file_sync,
            file_id
        )
    
    def _get_file_info_sync(self, file_id: str) -> Dict[str, Any]:
        """Get file information synchronously"""
        try:
            service = self._get_drive_service()
            file_info = service.files().get(
                fileId=file_id,
                fields='id,name,size,createdTime,modifiedTime,webViewLink,webContentLink'
            ).execute()
            
            return {
                'success': True,
                'file_info': file_info
            }
            
        except HttpError as error:
            return {
                'success': False,
                'error': f"Failed to get file info: {error}"
            }
    
    async def get_file_info_async(self, file_id: str) -> Dict[str, Any]:
        """Async wrapper for getting file information"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_file_info_sync,
            file_id
        )
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)