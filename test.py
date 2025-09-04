# import asyncio
# import httpx
# import time
# import json
# import os

# API_BASE_URL = "http://localhost:8000"
# # !!! IMPORTANT: Change this to a path of an actual, small video file on your computer !!!
# VIDEO_FILE_PATH = r"C:\Users\kisha\Downloads\windowsinstallation.mp4"

# # --- Helper Functions to Start Tasks ---

# async def start_process_request(client: httpx.AsyncClient, user_id: str):
#     """
#     Tests the /process-video/ endpoint which uses a server-side file path.
#     """
#     print(f"\n[TEST] Starting task for user '{user_id}' via /process-video/ endpoint...")
    
#     # Check if the file exists before making the request
#     if not os.path.exists(VIDEO_FILE_PATH):
#         print(f"[ERROR] Video file not found at: {VIDEO_FILE_PATH}. Cannot run /process-video/ test.")
#         return None

#     payload = {
#         "input_video_path": VIDEO_FILE_PATH,
#         "user_id": user_id,
#         "keep_intermediate_files": False,
#         "config": {"tts_engine": "gtts"} # Example config
#     }
    
#     try:
#         response = await client.post(f"{API_BASE_URL}/process-video/", json=payload, timeout=30)
#         response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
#         data = response.json()
#         task_id = data.get("task_id")
#         print(f"[SUCCESS] Started task via /process-video/. Task ID: {task_id}")
#         return task_id
#     except httpx.RequestError as e:
#         print(f"[ERROR] Request failed for /process-video/: {e}")
#     except Exception as e:
#         print(f"[ERROR] An unexpected error occurred: {e}")
#     return None

# async def start_upload_request(client: httpx.AsyncClient, user_id: str):
#     """
#     Tests the /upload-and-process/ endpoint which uploads a file.
#     """
#     print(f"\n[TEST] Starting task for user '{user_id}' via /upload-and-process/ endpoint...")
    
#     if not os.path.exists(VIDEO_FILE_PATH):
#         print(f"[ERROR] Video file not found at: {VIDEO_FILE_PATH}. Cannot run /upload-and-process/ test.")
#         return None

#     # Prepare data for multipart/form-data request
#     form_data = {
#         "user_id": user_id,
#         "keep_intermediate_files": str(False), # Booleans should be strings in form data
#         "config": json.dumps({"whisper_model_size": "tiny"}) # Config must be a JSON string
#     }

#     try:
#         with open(VIDEO_FILE_PATH, "rb") as video_file:
#             files = {"video_file": (os.path.basename(VIDEO_FILE_PATH), video_file, "video/mp4")}
#             response = await client.post(
#                 f"{API_BASE_URL}/upload-and-process/",
#                 data=form_data,
#                 files=files,
#                 timeout=60 # Allow more time for upload
#             )
#             response.raise_for_status()

#             data = response.json()
#             task_id = data.get("task_id")
#             print(f"[SUCCESS] Started task via /upload-and-process/. Task ID: {task_id}")
#             return task_id
#     except httpx.RequestError as e:
#         print(f"[ERROR] Request failed for /upload-and-process/: {e}")
#     except Exception as e:
#         print(f"[ERROR] An unexpected error occurred: {e}")
#     return None


# # --- Helper Function to Poll Task Status ---

# async def poll_task_status(client: httpx.AsyncClient, task_id: str):
#     """
#     Polls the status of a task until it's completed or failed.
#     """
#     print(f"[POLL] Starting to poll status for task: {task_id}")
#     start_time = time.time()
    
#     while True:
#         try:
#             response = await client.get(f"{API_BASE_URL}/status/{task_id}")
#             if response.status_code == 200:
#                 data = response.json()
#                 status = data.get("status")
#                 progress = data.get("progress")
                
#                 print(f"[POLL] Task {task_id}: Status is '{status}' - Progress: '{progress}'")

#                 if status in ["completed", "failed"]:
#                     end_time = time.time()
#                     print("-" * 50)
#                     print(f"[RESULT] Task {task_id} finished in {end_time - start_time:.2f} seconds.")
#                     print(f"Final Status: {data['status']}")
#                     if data['error']:
#                         print(f"Error: {data['error']}")
#                     if data['result']:
#                         # Pretty print the final result
#                         print("Result:", json.dumps(data['result'], indent=2))
#                     print("-" * 50)
#                     break
#             else:
#                 print(f"[POLL-ERROR] Failed to get status for {task_id}. Status Code: {response.status_code}")
#                 break

#         except httpx.RequestError as e:
#             print(f"[POLL-ERROR] Request failed while polling {task_id}: {e}")
#             break
        
#         await asyncio.sleep(5)  # Wait 5 seconds before polling again


# async def main():
#     all_task_ids = []
    
#     async with httpx.AsyncClient() as client:
#         # --- PHASE 1: START ALL TASKS ---
#         print("--- STARTING TASKS ---")
        
#         # Start two tasks concurrently: one for each endpoint
#         started_tasks = await asyncio.gather(
#             start_process_request(client, user_id="user_path_001"),
#             start_upload_request(client, user_id="user_upload_002")
#         )
        
#         # Collect the task IDs that were successfully created
#         all_task_ids = [task_id for task_id in started_tasks if task_id]

#         if not all_task_ids:
#             print("\nNo tasks were started. Exiting.")
#             return

#         # --- PHASE 2: POLL FOR RESULTS ---
#         print(f"\n--- POLLING STATUS FOR {len(all_task_ids)} TASKS ---")
        
#         # Create a list of polling tasks to run concurrently
#         polling_tasks = [poll_task_status(client, task_id) for task_id in all_task_ids]
#         await asyncio.gather(*polling_tasks)

#     print("\n--- TEST SCRIPT FINISHED ---")

# if __name__ == "__main__":
#     # Check for video file existence before running
#     if not os.path.exists(VIDEO_FILE_PATH):
#         print("="*60)
#         print("!!! FATAL ERROR !!!")
#         print(f"The video file specified in VIDEO_FILE_PATH does not exist.")
#         print(f"Path: {VIDEO_FILE_PATH}")
#         print("Please update the path and run the script again.")
#         print("="*60)
#     else:
#         asyncio.run(main())

# test_process_video_endpoint.py
import requests
import json
import os

# Your bearer token
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2OGI1ZjQxOWRhYTU5YzE0ZmE0Y2QzYWMiLCJlbWFpbCI6Imtpc2hhbnRyaXBhdGhpMjAyNUBnbWFpbC5jb20iLCJleHAiOjE3NTY5MzE0ODJ9.nVkOpfjuj14Sql2EQXyp2tG7GrK8nXXZK5UrFwjZFHQ"

# API endpoint
API_URL = "http://localhost:8000/process-video"

# Video file path
VIDEO_PATH = r"C:\Users\kisha\Downloads\hallucinations.mp4"

def test_process_video():
    """Simple test for /process-video endpoint"""
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return
    
    # Headers with your bearer token
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    # Optional parameters
    data = {
        "background_image_path": "",  # Optional
        "overlay_options": json.dumps({"opacity": 0.5})  # Optional JSON string
    }
    
    try:
        # Open video file
        with open(VIDEO_PATH, "rb") as video_file:
            files = {
                "file": (
                    os.path.basename(VIDEO_PATH),
                    video_file,
                    "video/mp4"
                )
            }
            
            print(f"�� Uploading video: {os.path.basename(VIDEO_PATH)}")
            print(f"📊 File size: {os.path.getsize(VIDEO_PATH) / (1024*1024):.2f} MB")
            
            # Make the request
            response = requests.post(
                API_URL,
                data=data,
                files=files,
                headers=headers,
                timeout=60
            )
            
            print(f"📡 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"🆔 Task ID: {result.get('task_id')}")
                print(f"📊 Status: {result.get('status')}")
                print(f"💬 Message: {result.get('message')}")
            else:
                print("❌ Failed!")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_process_video()


# #============================= test mongo db ==========================

# # test_mongodb_atlas.py
# import asyncio
# import os
# from motor.motor_asyncio import AsyncIOMotorClient
# from dotenv import load_dotenv

# load_dotenv()

# async def test_connection():
#     uri = os.getenv('MONGODB_URI')
    
#     try:
#         # Connect to Atlas
#         client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        
#         # Test connection
#         await client.admin.command('ping')
#         print("✅ MongoDB Atlas connection successful!")
        
#         # List databases
#         dbs = await client.list_database_names()
#         print(f"📁 Databases: {dbs}")
        
#         # Test write
#         db = client['video_processor']
#         result = await db.test.insert_one({"test": "data"})
#         print(f"✅ Write test successful: {result.inserted_id}")
        
#         # Test read
#         doc = await db.test.find_one({"test": "data"})
#         print(f"✅ Read test successful: {doc}")
        
#         # Cleanup
#         await db.test.delete_one({"test": "data"})
        
#         # Get cluster info
#         info = await client.server_info()
#         print(f"📊 MongoDB version: {info.get('version')}")
        
#     except Exception as e:
#         print(f"❌ Connection failed: {e}")
#     finally:
#         client.close()

# # Run test
# asyncio.run(test_connection())
