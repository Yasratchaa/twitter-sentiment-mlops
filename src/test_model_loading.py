# src/test_model_loading.py
import joblib
import json

def test_model():
    print("🧪 Testing model loading...")
    
    try:
        # Load model
        model = joblib.load('models/sentiment_model.pkl')
        print("✅ Model loaded successfully!")
        
        # Load metrics
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"📊 Model accuracy: {metrics['accuracy']:.4f}")
        
        # Test prediction
        test_text = "This is amazing!"
        prediction = model.predict([test_text])[0]
        confidence = model.predict_proba([test_text])[0].max()
        
        print(f"\n🎯 Test Prediction:")
        print(f"Text: '{test_text}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        
        print("\n✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_model()