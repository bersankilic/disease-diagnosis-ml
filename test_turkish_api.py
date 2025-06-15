#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe çeviri API'sini test eden script
"""

import requests
import json

# Test için bazı semptomları seçelim
test_symptoms = ['itching', 'skin_rash', 'high_fever', 'headache', 'nausea']

# API'ye tahmin isteği gönder
url = 'http://127.0.0.1:5000/predict'
data = {
    'symptoms': test_symptoms,
    'model': 'Random Forest'
}

try:
    response = requests.post(url, json=data)
    result = response.json()

    print("=== TAHMİN SONUCU ===")
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

# Türkçe semptom listesini de test edelim
print("\n=== TÜRKÇE SEMPTOM LİSTESİ TESTİ ===")
try:
    response = requests.get('http://127.0.0.1:5000/symptoms-turkish')
    symptoms = response.json()

    print(f"Toplam semptom sayısı: {len(symptoms)}")
    print("İlk 10 semptom:")
    for symptom in symptoms[:10]:
        print(f"  {symptom['english']} -> {symptom['turkish']}")

except Exception as e:
    print(f"Semptom listesi test hatası: {e}")
