# quick_test.py
import requests
import json

def quick_test():
    print("🧪 Quick Docker Container Test")
    print("=" * 40)
    
    # Test prediction
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": "I love Docker!"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    quick_test()