import os
import numpy as np
import tensorflow as tf
import keras
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model using Keras 3
model = keras.models.load_model('cotton_disease_model.keras')

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
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided in request'}), 400

        file = request.files['image']
        
        # Read the raw image bytes directly from memory
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        
        # Preprocess
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_batch, verbose=0)[0]
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
