from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import tempfile
import os
import cv2
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_age():
    tmp_file_path = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'success': False}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Check file size
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)  # Reset file pointer
        
        if file_size == 0:
            return jsonify({'error': 'Empty file', 'success': False}), 400
        
        # Convert uploaded file to OpenCV format
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format', 'success': False}), 400
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            tmp_file_path = tmp_file.name
        
        # Analyze the image
        app.logger.info("Starting DeepFace analysis...")
        result = DeepFace.analyze(
            img_path=tmp_file_path, 
            actions=['age'], 
            enforce_detection=False,
            detector_backend='opencv'  # Use opencv as it's more reliable
        )
        app.logger.info(f"DeepFace result: {result}")
        
        if result and len(result) > 0:
            age = result[0]['age']
            app.logger.info(f"Predicted age: {age}")
            
            return jsonify({
                'age': age,
                'success': True
            })
        else:
            return jsonify({'error': 'No face detected in the image', 'success': False}), 400
            
    except Exception as e:
        app.logger.error(f"Error in predict_age: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}', 
            'success': False
        }), 500
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up temp file: {e}")

if __name__ == '__main__':
    app.run(debug=True)