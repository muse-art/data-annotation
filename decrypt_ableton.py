#!/usr/bin/env python3

import sys
import gzip
import os

def decrypt_ableton_preset(input_path, output_path):
    """
    Decrypt Ableton preset files (.adg or .adv) by extracting the XML content.
    
    Args:
        input_path: Path to the .adg or .adv file
        output_path: Path where the extracted XML should be saved
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    try:
        # Read the gzipped content
        with gzip.open(input_path, 'rb') as f_in:
            xml_content = f_in.read()
        
        # Write the XML content to the output file
        with open(output_path, 'wb') as f_out:
            f_out.write(xml_content)
        
        print(f"Successfully decrypted '{input_path}' to '{output_path}'")
        return True
    
    except gzip.BadGzipFile:
        print(f"Error: '{input_path}' is not a valid gzip file.")
        return False
    except Exception as e:
        print(f"Error decrypting file: {str(e)}")
        return False

def main():
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python decrypt_ableton.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Add .xml extension if output doesn't have an extension
    if '.' not in os.path.basename(output_path):
        output_path += '.xml'
    
    success = decrypt_ableton_preset(input_path, output_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 