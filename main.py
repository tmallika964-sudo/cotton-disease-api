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

        # 2. Fallback: Check for JSON / base64 payload if no file stream was attached
        if not file_bytes:
            data = request.get_json(silent=True) or {}
            image_data = data.get('image') or data.get('file')
            if image_data and isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                file_bytes = base64.b64decode(image_data)

        if not file_bytes:
            return jsonify({'error': "No file uploaded"}), 400

        # 3. Open image and resize
        img = Image.open(BytesIO(file_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # 4. Convert to float32 and scale to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 5. Apply ImageNet Mean and Std Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Add batch dimension -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        # Predict via ONNX Engine
        predictions = session.run([output_name], {input_name: img_batch})[0][0]
        best_index = int(np.argmax(predictions))
        
        return jsonify({
            'class': class_names[best_index],
            'confidence': f"{float(predictions[best_index]) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
