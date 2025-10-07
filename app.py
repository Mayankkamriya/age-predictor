from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import tempfile
import os
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_age():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        
        image_file = request.files['image']
        
        # Convert uploaded file to OpenCV format
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            
            # Analyze the image
            result = DeepFace.analyze(img_path=tmp_file.name, actions=['age'], enforce_detection=False)
            age = result[0]['age']
            
            return jsonify({
                'age': age,
                'success': True
            })
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True)