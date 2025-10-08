from flask import Flask, request, jsonify, render_template
import tempfile
import os
import cv2
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

def estimate_age_from_face_size(face_area, image_area):
    """
    Simple age estimation based on face size relative to image
    This is a fallback when DeepFace is not available
    """
    face_ratio = face_area / image_area
    
    # Very rough estimation - larger faces (closer) tend to be adults
    if face_ratio > 0.3:
        return 35  # Adult
    elif face_ratio > 0.15:
        return 25  # Young adult
    elif face_ratio > 0.08:
        return 45  # Middle-aged
    else:
        return 55  # Senior

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
        
        # Check file size (limit to 5MB)
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size == 0:
            return jsonify({'error': 'Empty file', 'success': False}), 400
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            return jsonify({'error': 'File too large (max 5MB)', 'success': False}), 400
        
        # Convert to OpenCV format
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format', 'success': False}), 400
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            tmp_file_path = tmp_file.name
        
        # Try DeepFace first, fallback to OpenCV face detection
        try:
            from deepface import DeepFace
            app.logger.info("Using DeepFace for analysis...")
            result = DeepFace.analyze(
                img_path=tmp_file_path, 
                actions=['age'], 
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if result and len(result) > 0:
                age = result[0]['age']
                app.logger.info(f"DeepFace predicted age: {age}")
                
                return jsonify({
                    'age': age,
                    'success': True,
                    'method': 'deepface'
                })
                
        except Exception as deepface_error:
            app.logger.warning(f"DeepFace failed, using fallback: {deepface_error}")
        
        # Fallback: Simple face detection with OpenCV
        app.logger.info("Using OpenCV fallback...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face_area = w * h
            image_area = img.shape[0] * img.shape[1]
            
            estimated_age = estimate_age_from_face_size(face_area, image_area)
            
            return jsonify({
                'age': estimated_age,
                'success': True,
                'method': 'opencv_fallback',
                'message': 'Estimated using face detection'
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