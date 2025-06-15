#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost ve LightGBM modellerini test eden script
"""

import requests
import json

# Test için bazı semptomları seçelim
test_symptoms = ['itching', 'skin_rash', 'high_fever', 'headache', 'nausea']

# XGBoost modelini test et
print("=== XGBoost MODELİ TESTİ ===")
url = 'http://127.0.0.1:5000/predict'
data = {
    'symptoms': test_symptoms,
    'model': 'XGBoost'
}

try:
    response = requests.post(url, json=data)
    result = response.json()

    print(f"Seçilen semptomlar: {test_symptoms}")
    print(f"Tahmin edilen hastalık: {result.get('prediction', 'Bilinmiyor')}")
    print(f"Kullanılan model: {result.get('model_used', 'Bilinmiyor')}")

    if 'probabilities' in result and result['probabilities']:
        print("\nOlasılık dağılımı:")
        for disease, prob in result['probabilities']:
            print(f"  {disease}: {prob:.3f} ({prob*100:.1f}%)")

    if 'error' in result:
        print(f"Hata: {result['error']}")

except Exception as e:
    print(f"Test hatası: {e}")

print("\n" + "="*50 + "\n")

# LightGBM modelini test et
print("=== LightGBM MODELİ TESTİ ===")
data = {
    'symptoms': test_symptoms,
    'model': 'LightGBM'
}

try:
    response = requests.post(url, json=data)
    result = response.json()

    print(f"Seçilen semptomlar: {test_symptoms}")
    print(f"Tahmin edilen hastalık: {result.get('prediction', 'Bilinmiyor')}")
    print(f"Kullanılan model: {result.get('model_used', 'Bilinmiyor')}")

    if 'probabilities' in result and result['probabilities']:
        print("\nOlasılık dağılımı:")
        for disease, prob in result['probabilities']:
            print(f"  {disease}: {prob:.3f} ({prob*100:.1f}%)")

    if 'error' in result:
        print(f"Hata: {result['error']}")

except Exception as e:
    print(f"Test hatası: {e}")
