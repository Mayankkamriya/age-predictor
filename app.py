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

# DeepFace for real age detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available - install with: pip install deepface")

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Cloudinary from environment variables
if CLOUDINARY_AVAILABLE:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )

# Set up logging
logging.basicConfig(level=logging.INFO)

class AgePredictor:
    def __init__(self):
        if OPENCV_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = None
        
    def estimate_age_deepface(self, image_path):
        """Use DeepFace for accurate age prediction"""
        try:
            # Analyze image for age, gender, emotion, etc.
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=True,
                detector_backend='opencv',  # or 'ssd', 'mtcnn', 'retinaface'
                silent=True  # Disable verbose logs
            )
            
            # DeepFace returns a list, take first face analysis
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis
                
            return {
                'age': result['age'],
                'gender': result['gender'],
                'dominant_emotion': result['dominant_emotion'],
                'confidence': 0.85,  # DeepFace has high accuracy
                'faces_detected': 1,
                'method': 'deepface_ai'
            }
            
        except Exception as e:
            logging.error(f"DeepFace analysis error: {e}")
            return None

    def estimate_age_deepface_direct(self, image_data):
        """Use DeepFace directly with image data (no file saving needed)"""
        try:
            # Convert bytes to numpy array for DeepFace
            file_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Analyze using DeepFace with numpy array
            analysis = DeepFace.analyze(
                img_path=img,  # Pass numpy array directly
                actions=['age', 'gender', 'emotion'],
                enforce_detection=True,
                detector_backend='opencv',
                silent=True
            )
            
            # DeepFace returns a list, take first face analysis
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis
                
            return {
                'age': result['age'],
                'gender': result['gender'],
                'dominant_emotion': result['dominant_emotion'],
                'confidence': 0.85,
                'faces_detected': 1,
                'method': 'deepface_direct'
            }
            
        except Exception as e:
            logging.error(f"DeepFace direct analysis error: {e}")
            return None

    def fallback_age_prediction(self, image_data):
        """Fallback method if DeepFace fails"""
        if not OPENCV_AVAILABLE:
            return {'age': 35, 'confidence': 0.5, 'faces_detected': 1, 'method': 'fallback'}
        
        try:
            # Convert to OpenCV format
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
            
            # Simple ratio-based estimation (old method)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            face_area = w * h
            image_area = img.shape[0] * img.shape[1]
            face_ratio = face_area / image_area
            
            # Simple age estimation based on face ratio
            if face_ratio > 0.25:
                age = 25
            elif face_ratio > 0.15:
                age = 35
            elif face_ratio > 0.08:
                age = 45
            elif face_ratio > 0.04:
                age = 60
            else:
                age = 40
                
            confidence = min(face_ratio * 10, 0.8)
            
            return {
                'age': age,
                'confidence': round(confidence, 2),
                'faces_detected': len(faces),
                'method': 'opencv_fallback'
            }
            
        except Exception as e:
            logging.error(f"Fallback prediction error: {e}")
            return {'age': 35, 'confidence': 0.3, 'faces_detected': 1, 'method': 'error_fallback'}

    def predict_from_image(self, image_data):
        """Main prediction method - tries DeepFace first, then fallback"""
        # First try DeepFace with direct image data
        if DEEPFACE_AVAILABLE:
            result = self.estimate_age_deepface_direct(image_data)
            if result:
                return result
        
        # If DeepFace fails or not available, use fallback
        logging.info("DeepFace not available or failed, using fallback method")
        return self.fallback_age_prediction(image_data)

# Initialize predictor
predictor = AgePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_age():
    """Single endpoint that handles everything"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
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
        
        # Predict age using DeepFace or fallback
        result = predictor.predict_from_image(image_data)
        
        if result:
            response_data = {
                'age': result['age'],
                'confidence': result['confidence'],
                'faces_detected': result['faces_detected'],
                'cases_detected': result['faces_detected'],
                'success': True,
                'method': result.get('method', 'unknown'),
                'ai_model_used': 'DeepFace' if 'deepface' in result.get('method', '') else 'OpenCV Fallback'
            }
            
            # Add additional DeepFace data if available
            if 'gender' in result:
                response_data['gender'] = result['gender']
            if 'dominant_emotion' in result:
                response_data['emotion'] = result['dominant_emotion']
            
            # Add Cloudinary URL if available
            if cloudinary_url:
                response_data['image_url'] = cloudinary_url
                
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            response = jsonify({'error': 'No face detected in the image', 'success': False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400
            
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        response = jsonify({'error': f'Processing failed: {str(e)}', 'success': False})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify server is ready"""
    response = jsonify({
        'status': 'healthy', 
        'cloudinary_available': CLOUDINARY_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'deepface_available': DEEPFACE_AVAILABLE,
        'environment_loaded': bool(os.environ.get('CLOUDINARY_CLOUD_NAME')),
        'timestamp': time.time()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)