
# import asyncio
# import logging
# import os
# import sys
# from pathlib import Path
# from google_drive_uploader import GoogleDriveUploader

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('upload_log.txt')
#     ]
# )

# logger = logging.getLogger(__name__)

# async def test_single_upload():
#     """Test single video upload with retry logic"""
    
#     # Configuration - UPDATE THESE PATHS
#     SERVICE_ACCOUNT_PATH = r"C:\personal_projs\fastapi-demo\service_keys.json"
#     FOLDER_ID = None
#     VIDEO_FILE_PATH = r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4"
    
#     if not os.path.exists(SERVICE_ACCOUNT_PATH):
#         logger.error(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
#         print(f"\n❌ Please make sure your service account JSON file is at: {SERVICE_ACCOUNT_PATH}")
#         return
    
#     if not os.path.exists(VIDEO_FILE_PATH):
#         logger.error(f"Video file not found: {VIDEO_FILE_PATH}")
#         print(f"\n❌ Please make sure your video file is at: {VIDEO_FILE_PATH}")
#         return
    
#     try:
#         uploader = GoogleDriveUploader(
#             service_account_path=SERVICE_ACCOUNT_PATH,
#             folder_id=FOLDER_ID,
#             max_workers=1  # Single upload, no concurrency issues
#         )
        
#         video_name = f"single_test_upload_{Path(VIDEO_FILE_PATH).stem}.mp4"
#         result = await uploader.upload_video_async(
#             video_path=VIDEO_FILE_PATH,
#             video_name=video_name,
#             retry_count=3
#         )
        
#         if result['success']:
#             print("\n" + "="*50)
#             print("🎉 SINGLE UPLOAD SUCCESSFUL!")
#             print("="*50)
#             print(f"File Name: {result['file_name']}")
#             print(f"File ID: {result['file_id']}")
#             print(f"Attempts: {result.get('attempt', 1)}")
#             print(f"View Link: {result['web_view_link']}")
#             print("="*50)
#         else:
#             print(f"\n❌ Upload failed: {result['error']}")
            
#     except Exception as e:
#         logger.error(f"Single upload test error: {e}")

# async def test_sequential_uploads():
#     """Test sequential uploads (recommended approach)"""
    
#     SERVICE_ACCOUNT_PATH = r"C:\personal_projs\fastapi-demo\service_keys.json"
#     FOLDER_ID = None
    
#     # Use the same file multiple times for testing
#     VIDEO_FILES = [
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4",
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4",
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4"
#     ]
    
#     existing_files = [f for f in VIDEO_FILES if os.path.exists(f)]
    
#     if not existing_files:
#         logger.warning("No video files found for sequential upload test")
#         print(f"\n❌ No video files found. Please make sure files exist at:")
#         for f in VIDEO_FILES:
#             print(f"   {f}")
#         return
    
#     if not os.path.exists(SERVICE_ACCOUNT_PATH):
#         logger.error(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
#         print(f"\n❌ Please make sure your service account JSON file is at: {SERVICE_ACCOUNT_PATH}")
#         return
    
#     try:
#         uploader = GoogleDriveUploader(
#             service_account_path=SERVICE_ACCOUNT_PATH,
#             folder_id=FOLDER_ID,
#             max_workers=1  # Sequential uploads
#         )
        
#         print(f"\n🔄 Starting sequential upload of {len(existing_files)} videos...")
        
#         # Use the sequential upload method
#         results = await uploader.upload_videos_sequential(existing_files)
        
#         # Process results
#         successful = sum(1 for r in results if r.get('success'))
#         failed = len(results) - successful
        
#         print("\n" + "="*60)
#         print("SEQUENTIAL UPLOAD RESULTS")
#         print("="*60)
        
#         for i, result in enumerate(results):
#             if result.get('success'):
#                 attempts = result.get('attempt', 1)
#                 print(f"✅ Video {i+1}: {result['file_name']} (attempt {attempts})")
#                 print(f"   Link: {result['web_view_link']}")
#             else:
#                 print(f"❌ Video {i+1}: {result.get('error', 'Unknown error')}")
        
#         print("="*60)
#         print(f"Summary: {successful} successful, {failed} failed")
#         print("="*60)
        
#     except Exception as e:
#         logger.error(f"Sequential upload test error: {e}")

# async def test_limited_concurrent_uploads():
#     """Test concurrent uploads with proper limits"""
    
#     SERVICE_ACCOUNT_PATH = r"C:\personal_projs\fastapi-demo\service_keys.json"
#     FOLDER_ID = None
    
#     VIDEO_FILES = [
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4",
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4",
#         r"C:\personal_projs\fastapi-demo\ffmprgvid.mp4"
#     ]
    
#     existing_files = [f for f in VIDEO_FILES if os.path.exists(f)]
    
#     if not existing_files:
#         logger.warning("No video files found")
#         print(f"\n❌ No video files found. Please make sure files exist at:")
#         for f in VIDEO_FILES:
#             print(f"   {f}")
#         return
    
#     if not os.path.exists(SERVICE_ACCOUNT_PATH):
#         logger.error(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
#         print(f"\n❌ Please make sure your service account JSON file is at: {SERVICE_ACCOUNT_PATH}")
#         return
    
#     try:
#         # Limit to 2 concurrent uploads max
#         uploader = GoogleDriveUploader(
#             service_account_path=SERVICE_ACCOUNT_PATH,
#             folder_id=FOLDER_ID,
#             max_workers=2  # Limited concurrency
#         )
        
#         print(f"\n⚡ Starting limited concurrent upload of {len(existing_files)} videos...")
        
#         # Create upload tasks with delays
#         tasks = []
#         for i, video_path in enumerate(existing_files):
#             video_name = f"concurrent_upload_{i+1}_{Path(video_path).name}"
#             task = uploader.upload_video_async(
#                 video_path=video_path, 
#                 video_name=video_name,
#                 retry_count=3
#             )
#             tasks.append(task)
        
#         # Execute with proper error handling
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Process results
#         successful = 0
#         failed = 0
        
#         print("\n" + "="*60)
#         print("LIMITED CONCURRENT UPLOAD RESULTS")
#         print("="*60)
        
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 print(f"❌ Video {i+1}: Exception - {result}")
#                 failed += 1
#             elif result.get('success'):
#                 attempts = result.get('attempt', 1)
#                 print(f"✅ Video {i+1}: {result['file_name']} (attempt {attempts})")
#                 print(f"   Link: {result['web_view_link']}")
#                 successful += 1
#             else:
#                 print(f"❌ Video {i+1}: {result.get('error', 'Unknown error')}")
#                 failed += 1
        
#         print("="*60)
#         print(f"Summary: {successful} successful, {failed} failed")
#         print("="*60)
        
#     except Exception as e:
#         logger.error(f"Limited concurrent upload test error: {e}")

# def main():
#     """Main function with improved test options"""
    
#     print("Improved Google Drive Video Uploader Test")
#     print("="*45)
#     print("1. Single video upload (safest)")
#     print("2. Sequential uploads (recommended for multiple files)")
#     print("3. Limited concurrent uploads (2 max)")
#     print("4. Exit")
    
#     choice = input("\nSelect test option (1-4): ").strip()
    
#     if choice == "1":
#         print("\n🚀 Running single video upload test...")
#         asyncio.run(test_single_upload())
#     elif choice == "2":
#         print("\n🔄 Running sequential uploads test...")
#         asyncio.run(test_sequential_uploads())
#     elif choice == "3":
#         print("\n⚡ Running limited concurrent uploads test...")
#         asyncio.run(test_limited_concurrent_uploads())
#     elif choice == "4":
#         print("Goodbye! 👋")
#         sys.exit(0)
#     else:
#         print("Invalid choice. Please run again and select 1-4.")
#         return
    
#     another = input("\nWould you like to run another test? (y/N): ").strip().lower()
#     if another in ['y', 'yes']:
#         main()

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n\nOperation cancelled by user. Goodbye! 👋")
#         sys.exit(0)
#     except Exception as e:
#         logger.error(f"Unexpected error in main: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# --- Make sure the test script can find your uploader class ---
# This assumes 'google_drive_uploader.py' is in the same directory.
# If it's elsewhere, you might need to adjust the Python path.
try:
    from google_drive_uploader import GoogleDriveUploader
except ImportError:
    print("Error: Could not find 'google_drive_uploader.py'.")
    print("Please make sure this test script is in the same directory as your uploader file.")
    sys.exit(1)


# ==============================================================================
# --- CONFIG: PLEASE EDIT THESE VALUES ---
# ==============================================================================

# 1. Path to your Google Cloud service account JSON file.
#    Example: "path/to/your/credentials.json"
SERVICE_ACCOUNT_FILE = "service_keys.json"

# 2. The ID of the Google Drive folder where you want to upload the file.
#    This is the string of characters at the end of the folder's URL.
#    Example: "1a2b3c4d5e6f7g8h9i0j"
FOLDER_ID = "1LXXO5aarhVazOAZdLgryNWgvvxQ9mKa7"

# 3. The full path to the local video file you want to upload for this test.
#    Example: "/home/user/videos/test_video.mp4" or "C:\\Users\\User\\Videos\\test.mp4"
VIDEO_FILE_TO_UPLOAD = r"C:\Users\kisha\Downloads\Chatur_AI_HR_demo.mp4"

# ==============================================================================
# --- END OF CONFIG ---
# ==============================================================================


async def main():
    """
    The main function to run the uploader test.
    """
    print("--- Starting Google Drive Uploader Test ---")

    # --- Basic validation of config values ---
    if "path/to" in SERVICE_ACCOUNT_FILE or not os.path.exists(SERVICE_ACCOUNT_FILE):
        logging.error(f"Service account file not found or path not updated: {SERVICE_ACCOUNT_FILE}")
        return

    if "YOUR_GOOGLE_DRIVE_FOLDER_ID" in FOLDER_ID:
        logging.error("Please update the FOLDER_ID variable with your actual Google Drive folder ID.")
        return

    if "path/to" in VIDEO_FILE_TO_UPLOAD or not os.path.exists(VIDEO_FILE_TO_UPLOAD):
        logging.error(f"Video file for upload not found or path not updated: {VIDEO_FILE_TO_UPLOAD}")
        return

    try:
        # --- Initialize the uploader ---
        print("\n1. Initializing GoogleDriveUploader...")
        uploader = GoogleDriveUploader(
            service_account_path=SERVICE_ACCOUNT_FILE,
            folder_id=FOLDER_ID
        )
        print("   Initialization successful.")

        # --- Start the upload ---
        print(f"\n2. Starting upload for: {VIDEO_FILE_TO_UPLOAD}")
        upload_result = await uploader.upload_video_async(
            video_path=VIDEO_FILE_TO_UPLOAD
        )

        # --- Print the final result ---
        print("\n3. Upload process finished.")
        print("--- FINAL RESULT ---")
        if upload_result and upload_result.get('success'):
            logging.info("✅ Upload SUCCEEDED!")
            print(f"   File ID: {upload_result.get('file_id')}")
            print(f"   File Name: {upload_result.get('file_name')}")
            print(f"   WebView Link: {upload_result.get('web_view_link')}")
        else:
            logging.error("❌ Upload FAILED.")
            print(f"   Error message: {upload_result.get('error')}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the test: {e}", exc_info=True)

    finally:
        print("\n--- Test script finished. ---")


if __name__ == "__main__":
    # Set up basic logging to see the output from the uploader class
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    
    # Suppress noisy logs from underlying libraries to keep the output clean
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    
    # Run the asynchronous main function
    asyncio.run(main())



    
