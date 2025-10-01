# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime

def load_data():
    """Load dataset from CSV file"""
    print("📊 Loading data...")
    try:
        df = pd.read_csv('data/twitter_sentiment.csv')
        print(f"✅ Data loaded successfully: {len(df)} samples")
        print(f"📈 Sentiment distribution:")
        print(df['sentiment'].value_counts())
        return df
    except FileNotFoundError:
        print("❌ Dataset not found! Please run 'python src/create_mock_data.py' first")
        return None

def prepare_data(df):
    """Prepare data for training"""
    print("\n🔧 Preparing data...")
    
    # Check for missing values
    if df.isnull().sum().any():
        print("⚠️  Found missing values, cleaning...")
        df = df.dropna()
    
    # Basic text cleaning (optional, simple version)
    df['text'] = df['text'].str.lower()  # Convert to lowercase
    
    X = df['text']
    y = df['sentiment']
    
    print(f"✅ Data prepared: {len(X)} samples")
    return X, y

def create_model():
    """Create ML pipeline"""
    print("\n🤖 Creating ML pipeline...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,      # Increase features for better performance
            stop_words='english',   # Remove common English words
            ngram_range=(1, 2),     # Use unigrams and bigrams
            min_df=2,               # Ignore terms that appear in less than 2 documents
            max_df=0.95             # Ignore terms that appear in more than 95% of documents
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,          # Increase iterations for convergence
            C=1.0                   # Regularization parameter
        ))
    ])
    
    print("✅ Pipeline created successfully")
    return pipeline

def train_and_evaluate(pipeline, X, y):
    """Train model and evaluate performance"""
    print("\n🚀 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Testing samples: {len(X_test)}")
    
    # Train model
    pipeline.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    # Make predictions
    print("\n📈 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Test with sample predictions
    print(f"\n🧪 Sample Predictions:")
    test_samples = [
        "I love this product!",
        "This is terrible",
        "It's okay, nothing special"
    ]
    
    for sample in test_samples:
        pred = pipeline.predict([sample])[0]
        prob = pipeline.predict_proba([sample])[0].max()
        print(f"Text: '{sample}'")
        print(f"Prediction: {pred} (confidence: {prob:.3f})")
        print("-" * 50)
    
    return pipeline, {
        'accuracy': accuracy,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat()
    }

def save_model_and_metrics(pipeline, metrics):
    """Save trained model and metrics"""
    print("\n💾 Saving model and metrics...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/sentiment_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: metrics.json")
    
    # Save model info
    model_info = {
        'model_type': 'Logistic Regression with TF-IDF',
        'features': 'TF-IDF vectors (max_features=5000, ngram_range=(1,2))',
        'accuracy': metrics['accuracy'],
        'train_date': metrics['timestamp'],
        'model_file': model_path
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Model info saved to: models/model_info.json")

def main():
    """Main training function"""
    print("🚀 Starting ML Model Training Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Create model
    pipeline = create_model()
    
    # Train and evaluate
    trained_pipeline, metrics = train_and_evaluate(pipeline, X, y)
    
    # Save everything
    save_model_and_metrics(trained_pipeline, metrics)
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()