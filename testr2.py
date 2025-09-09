import os
import sys
import argparse
import uuid
from app.services.cloudflarer2_uploader import CloudflareR2Uploader
from app.config import get_settings

def test_r2_upload(video_path: str):
    """
    Tests the functionality of the CloudflareR2Uploader with a provided video file.

    This test performs the following steps:
    1.  Ensures that the required Cloudflare R2 settings are configured.
    2.  Uses the video file path provided by the user.
    3.  Instantiates the CloudflareR2Uploader.
    4.  Calls the `upload_file` method to upload the video to the R2 bucket.
    5.  Prints the public URL and the pre-signed download URL returned by the uploader.
    6.  Performs basic validation to ensure the returned URLs look correct.
    """
    print("Starting Cloudflare R2 video upload test...")

    # 1. Load settings and check for credentials
    try:
        settings = get_settings()
        if not all([
            settings.cloudflare_account_id,
            settings.cloudflare_access_key_id,
            settings.cloudflare_secret_access_key,
            settings.r2_bucket_name,
            settings.r2_public_domain
        ]):
            print("\n[ERROR] Cloudflare R2 environment variables are not fully configured.")
            print("Please ensure CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCESS_KEY_ID, etc., are set in your .env file.")
            return
    except Exception as e:
        print(f"\n[ERROR] Could not load settings. Make sure your .env file exists and is correct. Details: {e}")
        return

    # 2. Use the provided video file path
    print(f"Using video file: {video_path}")

    try:
        # 3. Instantiate the uploader
        print("Initializing CloudflareR2Uploader...")
        uploader = CloudflareR2Uploader()

        # 4. Upload the file
        base_filename = os.path.basename(video_path)
        # Create a unique object name in R2 to prevent overwriting files
        object_name = f"test_video_uploads/{uuid.uuid4()}-{base_filename}"
        print(f"Attempting to upload '{video_path}' to R2 bucket '{settings.r2_bucket_name}' as '{object_name}'...")
        
        upload_result = uploader.upload_file(video_path, object_name)
        
        print("\n--- UPLOAD SUCCESSFUL ---")
        
        # 5. Print the results
        public_url = upload_result.get("public_url")
        download_url = upload_result.get("download_url")
        
        print(f"Public URL: {public_url}")
        print(f"Download URL (expires in 1 hour): {download_url}")
        
        # 6. Validate the results
        assert public_url, "Public URL should not be empty."
        assert download_url, "Download URL should not be empty."
        assert object_name in public_url, "Object name should be in the public URL."
        assert "X-Amz-Signature" in download_url, "Download URL should be a pre-signed URL."
        
        print("\n[SUCCESS] Test completed successfully. URLs look valid.")

    except FileNotFoundError:
        print(f"\n[ERROR] The file was not found at the specified path: {video_path}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the upload test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cloudflare R2 video upload with a specific video file.")
    parser.add_argument("video_path", type=str, help="The full path to the video file you want to upload.")
    
    args = parser.parse_args()
    video_to_upload = args.video_path

    if not os.path.isfile(video_to_upload):
        print(f"[ERROR] The file path provided does not exist or is not a file: {video_to_upload}", file=sys.stderr)
        sys.exit(1)
        
    test_r2_upload(video_to_upload)

