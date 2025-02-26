import subprocess
import os

def convert_mpg_to_mp4(input_path, output_path):
    try:
        print(f"Converting {input_path} to {output_path}")
        
        # Add increased probe parameters
        result = subprocess.run([
            'ffmpeg', '-y',
            '-analyzeduration', '100000000',  # 100M
            '-probesize', '100000000',        # 100M
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # For testing
            output_path
        ], capture_output=True, text=True)
        
        # Check if successful
        if result.returncode == 0:
            print("Conversion successful!")
            return True
        else:
            print("Conversion failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# Test with one file
input_file = "dataset/train/speaking/GRID/1/001.mpg"
output_file = "test_output.mp4"

success = convert_mpg_to_mp4(input_file, output_file) 