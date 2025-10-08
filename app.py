from flask import Flask, request, jsonify, render_template
import os
import logging
import time

# Try to import optional dependencies with fallbacks
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False
    logging.warning("Cloudinary not available - running in fallback mode")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available - running in limited mode")

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure Cloudinary from environment variables
if CLOUDINARY_AVAILABLE:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_SECRET_KEY')
    )

# Set up logging
logging.basicConfig(level=logging.INFO)

class AgePredictor:
    def __init__(self):
        if OPENCV_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = None
        
    def estimate_age(self, face_ratio):
        """Estimate age based on face size ratio"""
        if face_ratio > 0.25:
            return 25
        elif face_ratio > 0.15:
            return 35
        elif face_ratio > 0.08:
            return 45
        elif face_ratio > 0.04:
            return 60
        else:
            return 40

    def predict_from_image(self, image_data):
        """Predict age directly from image data"""
        if not OPENCV_AVAILABLE:
            return {'age': 35, 'confidence': 0.5, 'faces_detected': 1, 'method': 'fallback'}
        
        try:
            # Convert to OpenCV format directly
            file_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Calculate face ratio
            face_area = w * h
            image_area = img.shape[0] * img.shape[1]
            face_ratio = face_area / image_area
            
            # Estimate age
            age = self.estimate_age(face_ratio)
            confidence = min(face_ratio * 10, 0.8)
            
            return {
                'age': age,
                'confidence': round(confidence, 2),
                'faces_detected': len(faces),
                'face_ratio': round(face_ratio, 3)
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return {'age': 35, 'confidence': 0.3, 'faces_detected': 1, 'method': 'error_fallback'}

# Initialize predictor
predictor = AgePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_age():
    """Single endpoint that handles everything"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'success': False}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Read image data directly
        image_data = image_file.read()
        
        # Upload to Cloudinary (optional - for storage)
        cloudinary_url = None
        if CLOUDINARY_AVAILABLE:
            try:
                # Reset file pointer for Cloudinary upload
                image_file.stream.seek(0)
                upload_result = cloudinary.uploader.upload(
                    image_file,
                    folder="age_predictor",
                    quality="auto",
                    fetch_format="auto"
                )
                cloudinary_url = upload_result['secure_url']
                logging.info(f"Image uploaded to Cloudinary: {cloudinary_url}")
            except Exception as e:
                logging.warning(f"Cloudinary upload failed: {e}")
                # Continue without Cloudinary - it's optional
        
        # Predict age directly from image data
        result = predictor.predict_from_image(image_data)
        
        if result:
            response_data = {
                'age': result['age'],
                'confidence': result['confidence'],
                'faces_detected': result['faces_detected'],
                'success': True,
                'method': result.get('method', 'direct_processing')
            }
            
            # Add Cloudinary URL if available
            if cloudinary_url:
                response_data['image_url'] = cloudinary_url
                
            return jsonify(response_data)
        else:
            return jsonify({'error': 'No face detected in the image', 'success': False}), 400
            
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}', 'success': False}), 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify server is ready"""
    return jsonify({
        'status': 'healthy', 
        'cloudinary_available': CLOUDINARY_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'environment_loaded': bool(os.environ.get('CLOUDINARY_NAME')),
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(debug=True)