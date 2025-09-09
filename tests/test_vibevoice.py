import requests
import tempfile
import os
import uuid
from typing import Optional

def generate_audio(text: str, speaker_name: str, output_file: Optional[str] = None) -> str:
    """
    Generate audio using VibeVoice TTS API
    
    Args:
        text (str): The text to convert to speech
        speaker_name (str): The speaker voice to use
        output_file (str, optional): Output file path. If None, generates a unique filename
    
    Returns:
        str: Path to the generated audio file
    """
    
    # API endpoint
    url = "https://fyzy94d0jqdy.share.zrok.io/generate-audio/"
    
    # Model path (constant as specified)
    model_path = "VibeVoice-1.5B"
    
    # Prepend "Speaker 1: " to the input text
    formatted_text = f"Speaker 1: {text}"
    
    # Generate unique ID for temporary file
    temp_id = str(uuid.uuid4())
    temp_txt_file = f"temp_{temp_id}.txt"
    
    try:
        # Create temporary text file with formatted text
        with open(temp_txt_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        # Prepare multipart form data
        with open(temp_txt_file, 'rb') as txt_file:
            files = {
                'txt_file': ('text.txt', txt_file, 'text/plain')
            }
            
            data = {
                'model_path': model_path,
                'speaker_names': [speaker_name]
            }
            
            # Make POST request
            response = requests.post(url, files=files, data=data)
            
            # Check if request was successful
            if response.status_code == 200:
                # Generate output filename if not provided
                if output_file is None:
                    output_file = f"output_{temp_id}.wav"
                
                # Save the audio file
                with open(output_file, 'wb') as audio_file:
                    audio_file.write(response.content)
                
                print(f"Audio generated successfully: {output_file}")
                return output_file
            else:
                raise Exception(f"API request failed with status code: {response.status_code}")
    
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        raise
    
    finally:
        # Clean up temporary text file
        if os.path.exists(temp_txt_file):
            os.remove(temp_txt_file)

def main():
    """
    Example usage of the generate_audio function
    """
    # Example text and speaker
    text = "Hello, this is a test of the VibeVoice text-to-speech system."
    speaker_name = "Alice_woman"  # Available speakers listed in comments
    
    try:
        # Generate audio
        output_path = generate_audio(text, speaker_name)
        print(f"Audio file saved to: {output_path}")
    
    except Exception as e:
        print(f"Failed to generate audio: {e}")

if __name__ == "__main__":
    # Available speakers:
    # ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman', 
    #  'en-Mary_woman_bgm', 'in-Samuel_man', 'zh-Anchen_man_bgm', 
    #  'zh-Bowen_man', 'zh-Xiran_woman']

    main()