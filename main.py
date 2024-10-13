from flask import Flask, request, render_template, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import io
from lime import lime_image
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the Keras models
model1 = load_model('BolneLgRhi.h5')
model2 = load_model('BolneLgRhiAdvance.h5')

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Preprocessing function for image prediction
def preprocess_image(image):
    image = np.array(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # If image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    resized_image = cv2.resize(image, (224, 224))  # Match size used during training
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Preprocessing function for LIME
def preprocess_image_for_lime(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        resized_img = img.resize((224, 224))  # Adjust size for LIME
        img_array = np.array(resized_img) / 255.0
        return np.expand_dims(img_array, axis=0), img.size
    except UnidentifiedImageError as e:
        print(f"Error processing image for LIME: {e}")
        return None, None

# Function to generate LIME explanations
def generate_lime_explanation(image_bytes):
    img_for_lime, original_size = preprocess_image_for_lime(image_bytes)

    if img_for_lime is None:
        return None, None, None

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_for_lime[0],
        lambda images: np.column_stack((model1.predict(images), model2.predict(images))),
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    label_to_explain = explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        label=label_to_explain,
        positive_only=True,
        num_features=5,
        hide_rest=True
    )

    # Upsample mask to original image size
    upsampled_mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

    # Find contours of highlighted regions
    contours, _ = cv2.findContours(upsampled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = [{"contour": [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in contour]} for contour in contours]

    return regions, contours, original_size

# Function to overlay shaded contours
def overlay_contours_on_image(image_bytes, contours, original_size):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, original_size)

    for contour in contours:
        cv2.fillPoly(img_resized, [contour], color=(0, 255, 0))  # Green fill

    alpha = 0.4  # Transparency factor
    shaded_img = cv2.addWeighted(img_resized, alpha, img_resized, 1 - alpha, 0)

    _, buffer = cv2.imencode('.png', shaded_img)  # Encode the shaded image as PNG
    return io.BytesIO(buffer)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
import base64
import io
import json
import os
from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file and file.filename != '':
        try:
            # Preprocess the image for prediction
            img = Image.open(io.BytesIO(file.read()))
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image file"}), 400

        img.seek(0)  # Reset the file pointer
        processed_image = preprocess_image(img)

        # Collect predictions from both Keras models
        preds1 = model1.predict(processed_image)[0][0]  # Binary output (sigmoid)
        preds2 = model2.predict(processed_image)[0][0]  # Binary output (sigmoid)

        # Prepare the features for XGBoost
        stacked_features = np.array([[preds1, preds2]])

        # Predict with the XGBoost model
        xgb_preds = xgb_model.predict(stacked_features)[0]

        # Convert NumPy types to native Python types
        is_forged = bool(xgb_preds >= 0.70)  # Convert to native bool
        confidence = float(xgb_preds) * 100 if is_forged else 100 - float(xgb_preds) * 100  # Convert to float

        # Proceed with LIME explanation
        file.seek(0)  # Reset file pointer again for LIME
        image_bytes = file.read()
        img_for_lime, original_size = preprocess_image_for_lime(image_bytes)

        if img_for_lime is None:
            return jsonify({"error": "Could not process image for LIME"}), 400

        # Generate LIME explanation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_for_lime[0],
            lambda images: np.column_stack((model1.predict(images), model2.predict(images))),
            top_labels=2,
            hide_color=0,
            num_samples=1000
        )

        # Get the top predicted label dynamically
        label_to_explain = explanation.top_labels[0]

        # Get the mask from LIME
        _, mask = explanation.get_image_and_mask(
            label=label_to_explain,
            positive_only=True,
            num_features=5,
            hide_rest=True
        )

        # Upsample mask to original image size
        upsampled_mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

        # Find contours of the highlighted regions in the mask
        contours, _ = cv2.findContours(upsampled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract full coordinates of each contour
        regions = []
        for contour in contours:
            contour_points = [{"x": int(point[0][0]), "y": int(point[0][1])} for point in contour]
            regions.append({"contour": contour_points})

        # Prepare the result with predictions and LIME explanation
        result = {
            "is_forged": is_forged,
            "confidence": confidence,
            "lime_explanation": regions
        }

        # Convert processed_image back to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Add the base64 image to the result
        result["image"] = f"data:image/jpeg;base64,{img_base64}"

        # Save the result to a local JSON file
        with open('prediction_result.json', 'w') as json_file:
            json.dump(result, json_file, indent=4)

        # Return the result as JSON
        return jsonify(result)

    return jsonify({"error": "Invalid file"}), 400


if __name__ == '__main__':
    app.run(debug=True)
