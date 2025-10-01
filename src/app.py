# src/app.py
from flask import Flask, request, jsonify
import joblib
import os
import json
from datetime import datetime
import traceback


app = Flask(__name__)

# Global variables
model = None
model_info = None

def load_model():
    """Load the trained model and info"""
    global model, model_info
    
    model_path = 'models/sentiment_model.pkl'
    info_path = 'models/model_info.json'
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Model loaded successfully!")
            
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                print("✅ Model info loaded successfully!")
            else:
                model_info = {"status": "Model info not found"}
                
            return True
        else:
            print("❌ Model file not found!")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running! 🚀',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': 'GET /',
            'predict': 'POST /predict',
            'model_info': 'GET /model-info',
            'batch_predict': 'POST /batch-predict'
        }
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model_info is None:
        return jsonify({'error': 'Model info not available'}), 500
    
    return jsonify({
        'model_info': model_info,
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required in JSON body'}), 400
        
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        # Make prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = probabilities.max()
        
        # Get all class probabilities
        classes = model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        return jsonify({
            'text': text,
            'sentiment': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'texts field is required in JSON body'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                return jsonify({'error': f'Text at index {i} must be a non-empty string'}), 400
        
        # Make predictions
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        
        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            confidence = probs.max()
            classes = model.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            
            results.append({
                'index': i,
                'text': text,
                'sentiment': pred,
                'confidence': float(confidence),
                'probabilities': prob_dict
            })
        
        return jsonify({
            'results': results,
            'total_predictions': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'health': 'GET /',
            'predict': 'POST /predict',
            'model_info': 'GET /model-info',
            'batch_predict': 'POST /batch-predict'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500


@app.route('/web')
def web_interface():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API...")
    print("=" * 50)
    
    # Load model on startup
    if load_model():
        print("🎯 Model loaded successfully!")
    else:
        print("⚠️  Model not loaded - API will return errors")
    
    print("🌐 Starting Flask server...")
    print("📡 API will be available at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8000, debug=True)