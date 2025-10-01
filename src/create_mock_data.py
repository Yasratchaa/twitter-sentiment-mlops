# src/create_mock_data.py
import pandas as pd
import numpy as np
import os

def create_mock_dataset():
    # Pastikan folder data ada
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Mock data untuk demo
    positive_texts = [
        "I love this product! Amazing quality!",
        "Excellent customer support, very helpful",
        "Great value for the price, highly recommend",
        "Outstanding performance, exceeded expectations",
        "Best purchase I've made this year",
        "Fantastic quality, will buy again",
        "Perfect service, very satisfied",
        "Amazing experience, thank you so much",
        "Love it! Works perfectly",
        "Great product, fast delivery"
    ]
    
    negative_texts = [
        "This is the worst service ever experienced",
        "Terrible experience, very disappointed",
        "Not worth the money, poor quality",
        "Poor quality, broke immediately after use",
        "Worst customer service, no response",
        "Completely useless, waste of money",
        "Very bad experience, would not recommend",
        "Terrible product, doesn't work as advertised",
        "Poor quality control, arrived damaged",
        "Disappointing purchase, regret buying"
    ]
    
    neutral_texts = [
        "Pretty good, would recommend to others",
        "It's okay, nothing special but works",
        "Average product, meets basic expectations",
        "Decent quality for the price range",
        "Not bad, could be better though",
        "Mediocre at best, average performance",
        "It works fine, nothing extraordinary",
        "Acceptable quality, standard service",
        "Fair product, reasonable price point",
        "Standard quality, as expected from brand"
    ]
    
    # Buat dataset dengan replikasi
    texts = []
    sentiments = []
    
    # Replikasi data untuk membuat dataset lebih besar
    for _ in range(50):  # 50x replikasi = 1500 total samples
        texts.extend(positive_texts)
        sentiments.extend(['positive'] * len(positive_texts))
        
        texts.extend(negative_texts)
        sentiments.extend(['negative'] * len(negative_texts))
        
        texts.extend(neutral_texts)
        sentiments.extend(['neutral'] * len(neutral_texts))
    
    # Buat DataFrame
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments
    })
    
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/twitter_sentiment.csv', index=False)
    
    print(f"✅ Dataset created successfully!")
    print(f"📊 Total samples: {len(df)}")
    print(f"📈 Sentiment distribution:")
    print(df['sentiment'].value_counts())
    print(f"💾 Saved to: data/twitter_sentiment.csv")

if __name__ == "__main__":
    create_mock_dataset()