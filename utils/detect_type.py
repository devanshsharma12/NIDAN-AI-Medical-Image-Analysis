# utils/detect_type.py

import os

def detect_image_type(image_path):
    """
    Detect image type based on filename. 
    Looks for 'eye' or 'brain' in the filename.
    """
    filename = os.path.basename(image_path).lower()

    if "eye" in filename:
        return "eye"
    elif "brain" in filename:
        return "brain"
    else:
        return "unknown"

