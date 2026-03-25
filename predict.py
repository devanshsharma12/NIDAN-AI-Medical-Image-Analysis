# # predict.py

import os
import numpy as np
import tensorflow as tf
import cv2

from utils.detect_type import detect_image_type
from utils.decode_result import decode_eye_result
from utils.preprocess import preprocess_eye_image

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
    return bce + (1 - dice)

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

@tf.keras.utils.register_keras_serializable()
def iou_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# Load models
eye_model = tf.keras.models.load_model("models/my_model.keras", compile=False)

brain_model = tf.keras.models.load_model(
    "models/best_brain_tumor_model.keras",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score
    }
)

# Detect input shapes
eye_input_shape = eye_model.input_shape[1:4]
brain_input_shape = brain_model.input_shape[1:4]

def preprocess_image(image_path, target_shape):
    H, W, C = target_shape

    if C == 1:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (W, H))

    if C == 1:
        img = np.expand_dims(img, axis=-1)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image_path):

    image_type = detect_image_type(os.path.basename(image_path))

    if image_type == "eye":

        image = preprocess_eye_image(image_path, target_size=eye_input_shape[:2])
        prediction = eye_model.predict(image)

        predicted_class = np.argmax(prediction)
        result = decode_eye_result(predicted_class)

        return result, None


    elif image_type == "brain":

        image = preprocess_image(image_path, brain_input_shape)
        prediction = brain_model.predict(image)[0]

        # Ensure mask is 2D
        if len(prediction.shape) == 3:
            prediction = prediction[:, :, 0]

        original = cv2.imread(image_path)

        if original is None:
            raise ValueError(f"Could not read original image: {image_path}")

        H, W = original.shape[:2]

        mask = cv2.resize(prediction, (W, H))
        mask = (mask > 0.5).astype(np.uint8) * 255

        tumor_present = np.sum(mask) > 0
        tumor_result = "Tumor detected" if tumor_present else "No tumor detected"

        overlay = original.copy()

        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 2] = mask

        overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.5, 0)

        # Ensure folder exists
        os.makedirs("static/uploads", exist_ok=True)

        original_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(original_filename)

        segmented_filename = f"{name}_segmented.png"
        segmented_path = os.path.join("static", "uploads", segmented_filename)

        cv2.imwrite(segmented_path, overlay)

        return tumor_result, f"uploads/{segmented_filename}"


    else:
        return "Unknown image type", None

# import os
# import numpy as np
# import tensorflow as tf
# import cv2

# from utils.detect_type import detect_image_type
# from utils.decode_result import decode_eye_result, decode_brain_result
# from utils.preprocess import preprocess_eye_image, preprocess_brain_image

# @tf.keras.utils.register_keras_serializable()
# def bce_dice_loss(y_true, y_pred):
#     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     dice = (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
#     return bce + (1 - dice)

# @tf.keras.utils.register_keras_serializable()
# def dice_coefficient(y_true, y_pred):
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

# @tf.keras.utils.register_keras_serializable()
# def iou_score(y_true, y_pred):
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#     return (intersection + 1e-7) / (union + 1e-7)

# # Load models
# eye_model = tf.keras.models.load_model("models/my_model.keras", compile=False)
# brain_model = tf.keras.models.load_model(
#     "models/best_brain_tumor_model.keras",
#     custom_objects={
#         "bce_dice_loss": bce_dice_loss,
#         "dice_coefficient": dice_coefficient,
#         "iou_score": iou_score
#     }
# )

# # Auto-detect input shape for each model
# eye_input_shape = eye_model.input_shape[1:4]   # (H, W, C)
# brain_input_shape = brain_model.input_shape[1:4]

# def preprocess_image(image_path, target_shape):
#     H, W, C = target_shape

#     if C == 1:
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     else:
#         img = cv2.imread(image_path)

#     if img is None:
#         raise ValueError(f"Could not read image: {image_path}")

#     img = cv2.resize(img, (W, H))

#     if C == 1:
#         img = np.expand_dims(img, axis=-1)

#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)

#     return img

# def predict_image(image_path):
#     image_type = detect_image_type(os.path.basename(image_path))

#     if image_type == "eye":
#         image = preprocess_eye_image(image_path, target_size=eye_input_shape[:2])
#         prediction = eye_model.predict(image)
#         predicted_class = np.argmax(prediction)
#         result = decode_eye_result(predicted_class)
#         return result, None

#     elif image_type == "brain":
#         image = preprocess_image(image_path, brain_input_shape)
#         prediction = brain_model.predict(image)[0]

#         # Resize mask back to original image shape
#         original = cv2.imread(image_path)
#         H, W = original.shape[:2]

#         mask = cv2.resize(prediction, (W, H))
#         mask = (mask > 0.5).astype(np.uint8) * 255

#         # Check for tumor presence
#         tumor_present = np.sum(mask) > 0
#         tumor_result = "Tumor detected" if tumor_present else "No tumor detected"

#         # Overlay mask on original image
#         overlay = original.copy()
#         if len(mask.shape) == 2:
#             mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         else:
#             mask_colored = mask

#         red_mask = np.zeros_like(overlay)
#         red_mask[:, :, 2] = mask
#         overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.5, 0)        # Save the segmented image with a unique name based on original image
#         original_filename = os.path.basename(image_path)
#         name, ext = os.path.splitext(original_filename)
#         segmented_filename = f"{name}_segmented.png"
#         segmented_path = os.path.join("static", "uploads", segmented_filename)
#         cv2.imwrite(segmented_path, overlay)

#         return tumor_result, f"uploads/{segmented_filename}"  # Return relative path for url_for

#     else:
#         return "Unknown image type", None

# # python predict.py path_to_image.jpg