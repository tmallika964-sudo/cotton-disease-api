import os
import numpy as np
import onnxruntime as ort
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

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
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided in request'}), 400

        file = request.files['image']
        
        # Preprocess image
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict via ONNX Engine
        predictions = session.run([output_name], {input_name: img_batch})[0][0]
        best_index = np.argmax(predictions)
        
        return jsonify({
            'class': class_names[best_index],
            'confidence': f"{float(predictions[best_index]) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
