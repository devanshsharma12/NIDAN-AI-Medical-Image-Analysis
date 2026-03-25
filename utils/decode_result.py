# utils/decode_result.py

# 
# Updated Eye disease class names for classification output
EYE_CLASSES = [
    "Normal (N)",
    "Diabetes (D)",
    "Glaucoma (G)",
    "Cataract (C)",
    "Age-related Macular Degeneration (A)",
    "Hypertension (H)",
    "Pathological Myopia (M)",
    "Other diseases/abnormalities (O)"
]

# Brain tumor classification
BRAIN_CLASSES = [
    "No Tumor",
    "Tumor Detected"
]

def decode_eye_result(predicted_class):
    """Return a string label for eye disease classification."""
    if 0 <= predicted_class < len(EYE_CLASSES):
        return f"Eye Condition: {EYE_CLASSES[predicted_class]}"
    else:
        return "Eye Condition: Unknown"

def decode_brain_result(predicted_class):
    """Return a string label for brain classification (if applicable)."""
    if 0 <= predicted_class < len(BRAIN_CLASSES):
        return f"Brain Tumor: {BRAIN_CLASSES[predicted_class]}"
    else:
        return "Brain Tumor: Unknown"