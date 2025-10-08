from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cv2
import numpy as np
import requests
import logging
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)

# Configure Cloudinary from environment variables (SECURE)
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_SECRET_KEY')
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Validate Cloudinary configuration
def validate_cloudinary_config():
    required_vars = ['CLOUDINARY_NAME', 'CLOUDINARY_API_KEY', 'CLOUDINARY_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logging.error(f"Missing Cloudinary environment variables: {', '.join(missing_vars)}")
        return False
    return True

class CloudinaryAgePredictor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def estimate_age(self, face_ratio, face_features=None):
        """Estimate age based on face size ratio"""
        if face_ratio > 0.25:
            return 25  # Large face (close up) - young adult
        elif face_ratio > 0.15:
            return 35  # Medium face - adult
        elif face_ratio > 0.08:
            return 45  # Smaller face - middle aged
        elif face_ratio > 0.04:
            return 60  # Very small face - senior
        else:
            return 40  # Default

    def predict_from_url(self, image_url):
        """Predict age from Cloudinary image URL"""
        try:
            # Download image from Cloudinary
            response = requests.get(image_url)
            if response.status_code != 200:
                return None
            
            # Convert to OpenCV format
            file_bytes = np.frombuffer(response.content, np.uint8)
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
            return None

# Initialize predictor
predictor = CloudinaryAgePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload image to Cloudinary and return URL"""
    # Check if Cloudinary is configured
    if not validate_cloudinary_config():
        return jsonify({'error': 'Server configuration error', 'success': False}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'success': False}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_file,
            folder="age_predictor",
            quality="auto",
            fetch_format="auto"
        )
        
        return jsonify({
            'success': True,
            'image_url': upload_result['secure_url'],
            'public_id': upload_result['public_id']
        })
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}', 'success': False}), 500

@app.route('/predict', methods=['POST'])
def predict_age():
    """Predict age from Cloudinary image URL"""
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image URL provided', 'success': False}), 400
        
        image_url = data['image_url']
        
        # Predict age using Cloudinary URL
        result = predictor.predict_from_url(image_url)
        
        if result:
            return jsonify({
                'age': result['age'],
                'confidence': result['confidence'],
                'faces_detected': result['faces_detected'],
                'success': True,
                'method': 'cloudinary_opencv'
            })
        else:
            return jsonify({'error': 'No face detected in the image', 'success': False}), 400
            
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_image():
    """Delete image from Cloudinary after processing"""
    # Check if Cloudinary is configured
    if not validate_cloudinary_config():
        return jsonify({'error': 'Server configuration error', 'success': False}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'public_id' not in data:
            return jsonify({'error': 'No public ID provided', 'success': False}), 400
        
        public_id = data['public_id']
        
        # Delete from Cloudinary
        result = cloudinary.uploader.destroy(public_id)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}', 'success': False}), 500

@app.route('/health')
def health_check():
    cloudinary_configured = validate_cloudinary_config()
    return jsonify({
        'status': 'healthy', 
        'using_cloudinary': True,
        'cloudinary_configured': cloudinary_configured
    })

if __name__ == '__main__':
    app.run(debug=True)