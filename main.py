import os
import io
import base64
import numpy as np
import onnxruntime as ort
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load ONNX model into memory
session = ort.InferenceSession('cotton_disease_model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

class_names = [
    'Bacterial Blight',
    'Curl Virus',
    'Healthy Leaf',
    'Leaf Redding',
    'Leaf Variegation'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file_bytes = None

        # 1. Check for standard multipart file upload ('image' or 'file' key)
        file = request.files.get('image') or request.files.get('file')
        if file:
            file_bytes = file.read()

        # 2. Fallback: Check for JSON / base64 payload
        if not file_bytes:
            data = request.get_json(silent=True) or {}
            image_data = data.get('image') or data.get('file')
            if image_data and isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                file_bytes = base64.b64decode(image_data)

        if not file_bytes:
            return jsonify({'error': "No file uploaded"}), 400

        print(f"DEBUG - Received file size: {len(file_bytes)} bytes", flush=True)

        # 3. Open and preprocess image
        img = Image.open(BytesIO(file_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        print(f"DEBUG - Pixel sample: {img_array[0, 0, :]}", flush=True)

        img_batch = np.expand_dims(img_array, axis=0)

        # 4. Predict via ONNX Engine
        predictions = session.run([output_name], {input_name: img_batch})[0][0]
        
        # Calculate Softmax probabilities to prevent invalid confidence percentages
        exp_preds = np.exp(predictions - np.max(predictions))
        probs = exp_preds / np.sum(exp_preds)
        
        best_index = int(np.argmax(probs))
        confidence_val = float(probs[best_index]) * 100

        print(f"DEBUG - Probabilities: {probs}", flush=True)

        return jsonify({
            'class': class_names[best_index],
            'confidence': f"{confidence_val:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
