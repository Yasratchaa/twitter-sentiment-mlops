# src/test_api.py
import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Sentiment Analysis API")
    print("=" * 50)
    
    # Wait a bit for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    try:
        # Test 1: Health Check
        print("\n1️⃣ Testing Health Check...")
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test 2: Model Info
        print("\n2️⃣ Testing Model Info...")
        response = requests.get(f"{base_url}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test 3: Single Predictions
        print("\n3️⃣ Testing Single Predictions...")
        test_cases = [
            "I absolutely love this product! It's amazing!",
            "This is the worst thing I've ever bought",
            "It's okay, nothing special but works fine",
            "Fantastic quality and great customer service!",
            "Terrible experience, would not recommend"
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n  Test {i}:")
            response = requests.post(
                f"{base_url}/predict",
                json={"text": text}
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Text: '{result['text']}'")
                print(f"  Sentiment: {result['sentiment']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Probabilities: {result['probabilities']}")
            else:
                print(f"  Error: {response.json()}")
        
        # Test 4: Batch Prediction
        print("\n4️⃣ Testing Batch Predictions...")
        batch_texts = [
            "Great product!",
            "Not good at all",
            "Average quality"
        ]
        
        response = requests.post(
            f"{base_url}/batch-predict",
            json={"texts": batch_texts}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total predictions: {result['total_predictions']}")
            for pred in result['results']:
                print(f"  '{pred['text']}' → {pred['sentiment']} ({pred['confidence']:.3f})")
        else:
            print(f"Error: {response.json()}")
        
        # Test 5: Error Handling
        print("\n5️⃣ Testing Error Handling...")
        
        # Empty text
        response = requests.post(
            f"{base_url}/predict",
            json={"text": ""}
        )
        print(f"Empty text - Status: {response.status_code}")
        
        # Missing field
        response = requests.post(
            f"{base_url}/predict",
            json={"wrong_field": "test"}
        )
        print(f"Missing field - Status: {response.status_code}")
        
        # Invalid endpoint
        response = requests.get(f"{base_url}/invalid-endpoint")
        print(f"Invalid endpoint - Status: {response.status_code}")
        
        print("\n✅ API testing completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running!")
        print("💡 Start the server with: python src/app.py")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_api()