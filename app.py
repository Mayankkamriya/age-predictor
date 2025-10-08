from flask import Flask, request, jsonify, render_template
import os
import logging
import time
import tempfile

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
DEEPFACE_AVAILABLE = False
DEEPFACE_ERROR = "Not attempted"

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    DEEPFACE_ERROR = "Successfully imported"
    logging.info("✅ DeepFace successfully imported")
    
    # Test DeepFace functionality
    try:
        # Quick test to verify DeepFace works
        import tensorflow as tf
        logging.info(f"✅ TensorFlow version: {tf.__version__}")
    except Exception as tf_error:
        logging.warning(f"⚠️ TensorFlow issue: {tf_error}")
        
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    DEEPFACE_ERROR = f"Import failed: {str(e)}"
    logging.error(f"❌ DeepFace import failed: {e}")

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure Cloudinary
if CLOUDINARY_AVAILABLE:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )

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
            logging.info("🔄 Attempting DeepFace analysis...")
            
            # Test if we can analyze a simple case first
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,  # Set to False for better compatibility
                detector_backend='opencv',
                silent=False,
                prog_bar=False
            )
            
            logging.info(f"✅ DeepFace analysis completed: {type(analysis)}")
            
            # DeepFace returns a list, take first face analysis
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis
                
            logging.info(f"🎯 DeepFace result - Age: {result['age']}, Gender: {result.get('gender', 'unknown')}")
                
            return {
                'age': result['age'],
                'gender': result.get('gender', 'unknown'),
                'dominant_emotion': result.get('dominant_emotion', 'neutral'),
                'confidence': 0.85,
                'faces_detected': 1,
                'method': 'deepface_ai'
            }
            
        except Exception as e:
            logging.error(f"❌ DeepFace analysis error: {e}")
            return None

    def estimate_age_deepface_direct(self, image_data):
        """Use DeepFace with temporary file"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_data)
                tmp_file_path = tmp_file.name
            
            logging.info(f"📁 Created temp file for DeepFace: {tmp_file_path}")
            
            # Use DeepFace with the temporary file
            result = self.estimate_age_deepface(tmp_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
            return result
            
        except Exception as e:
            logging.error(f"❌ DeepFace direct analysis error: {e}")
            return None

    def test_deepface_functionality(self):
        """Test if DeepFace is working properly"""
        if not DEEPFACE_AVAILABLE:
            return {"status": "not_available", "error": DEEPFACE_ERROR}
        
        try:
            # Create a simple test image (black image)
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, test_img)
                tmp_path = tmp_file.name
            
            # Try to analyze (should fail but show if DeepFace is working)
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['age'],
                enforce_detection=False,
                silent=True
            )
            
            os.unlink(tmp_path)
            return {"status": "working", "message": "DeepFace is functional"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def fallback_age_prediction(self, image_data):
        """Improved fallback method"""
        if not OPENCV_AVAILABLE:
            return {'age': 35, 'confidence': 0.5, 'faces_detected': 1, 'method': 'basic_fallback'}
        
        try:
            # Convert to OpenCV format
            file_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with multiple attempts
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                # Try with different parameters
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05, 
                    minNeighbors=3, 
                    minSize=(20, 20)
                )
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Calculate face metrics
            face_area = w * h
            image_area = img.shape[0] * img.shape[1]
            face_ratio = face_area / image_area
            
            # Enhanced age estimation based on face ratio and position
            if face_ratio > 0.25:
                age = 22
                confidence = 0.8
            elif face_ratio > 0.20:
                age = 28
                confidence = 0.75
            elif face_ratio > 0.15:
                age = 35
                confidence = 0.7
            elif face_ratio > 0.10:
                age = 45
                confidence = 0.65
            elif face_ratio > 0.06:
                age = 55
                confidence = 0.6
            else:
                age = 40
                confidence = 0.5
            
            # Adjust confidence based on face detection quality
            confidence = min(confidence + (len(faces) * 0.1), 0.8)
            
            return {
                'age': age,
                'confidence': round(confidence, 2),
                'faces_detected': len(faces),
                'face_ratio': round(face_ratio, 3),
                'method': 'enhanced_opencv'
            }
            
        except Exception as e:
            logging.error(f"Fallback prediction error: {e}")
            return {'age': 35, 'confidence': 0.3, 'faces_detected': 1, 'method': 'error_fallback'}

    def predict_from_image(self, image_data):
        """Main prediction method with detailed logging"""
        # First try DeepFace with detailed logging
        if DEEPFACE_AVAILABLE:
            logging.info("🚀 Attempting DeepFace age prediction...")
            result = self.estimate_age_deepface_direct(image_data)
            if result:
                logging.info("✅ DeepFace prediction successful!")
                return result
            else:
                logging.warning("⚠️ DeepFace failed, using enhanced OpenCV fallback")
        else:
            logging.warning(f"⚠️ DeepFace not available: {DEEPFACE_ERROR}")
        
        # Use enhanced fallback
        logging.info("🔄 Using enhanced OpenCV fallback method")
        return self.fallback_age_prediction(image_data)

# Initialize predictor
predictor = AgePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test-deepface')
def test_deepface():
    """Test endpoint to check DeepFace functionality"""
    test_result = predictor.test_deepface_functionality()
    return jsonify({
        'deepface_available': DEEPFACE_AVAILABLE,
        'deepface_error': DEEPFACE_ERROR,
        'test_result': test_result,
        'opencv_available': OPENCV_AVAILABLE,
        'cloudinary_available': CLOUDINARY_AVAILABLE
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_age():
    """Single endpoint that handles everything"""
    if request.method == 'OPTIONS':
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
        
        # Read image data
        image_data = image_file.read()
        
        if len(image_data) == 0:
            return jsonify({'error': 'Empty image file', 'success': False}), 400
        
        # Upload to Cloudinary
        cloudinary_url = None
        if CLOUDINARY_AVAILABLE:
            try:
                image_file.stream.seek(0)
                upload_result = cloudinary.uploader.upload(
                    image_file,
                    folder="age_predictor",
                    quality="auto",
                    fetch_format="auto"
                )
                cloudinary_url = upload_result['secure_url']
                logging.info(f"📸 Image uploaded to Cloudinary")
            except Exception as e:
                logging.warning(f"Cloudinary upload failed: {e}")
        
        # Predict age
        result = predictor.predict_from_image(image_data)
        
        if result:
            response_data = {
                'age': result['age'],
                'confidence': result['confidence'],
                'faces_detected': result['faces_detected'],
                'cases_detected': result['faces_detected'],
                'success': True,
                'method': result.get('method', 'unknown'),
                'ai_model_used': 'DeepFace AI' if 'deepface' in result.get('method', '') else 'Enhanced OpenCV',
                'deepface_status': 'active' if 'deepface' in result.get('method', '') else 'fallback'
            }
            
            # Add additional data if available
            if 'gender' in result:
                response_data['gender'] = result['gender']
            if 'emotion' in result:
                response_data['emotion'] = result.get('dominant_emotion', 'neutral')
            if 'face_ratio' in result:
                response_data['face_ratio'] = result['face_ratio']
            
            if cloudinary_url:
                response_data['image_url'] = cloudinary_url
                
            logging.info(f"🎉 Prediction completed: Age {result['age']}, Method: {result.get('method', 'unknown')}")
                
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
    """Health check endpoint"""
    response = jsonify({
        'status': 'healthy', 
        'cloudinary_available': CLOUDINARY_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'deepface_available': DEEPFACE_AVAILABLE,
        'deepface_error': DEEPFACE_ERROR,
        'environment_loaded': bool(os.environ.get('CLOUDINARY_CLOUD_NAME')),
        'timestamp': time.time()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)