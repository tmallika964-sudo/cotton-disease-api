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
        
       # 1. Open image and resize
img = Image.open(BytesIO(file.read())).convert('RGB')
img = img.resize((224, 224))

# 2. Convert to float32 and normalize
img_array = np.array(img, dtype=np.float32) / 255.0

# --- ADD THESE 2 CORRECTION LINES BELOW ---

# Convert RGB to BGR (matches OpenCV if trained in Colab)
img_array = img_array[:, :, ::-1] 

# Transpose from (224, 224, 3) to (3, 224, 224) (matches PyTorch/ONNX NCHW format)
img_array = np.transpose(img_array, (2, 0, 1)) 

# ------------------------------------------

# Add batch dimension -> becomes (1, 3, 224, 224)
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
