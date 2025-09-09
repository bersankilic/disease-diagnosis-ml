"""
Web API test scripti - Flask uygulamasının API endpoint'lerini test eder
"""

import requests
import json

def test_api():
    """API endpoint'lerini test et"""
    base_url = "http://localhost:5000"

    print("AI Hastalık Teşhis Sistemi API Testi")
    print("=" * 50)

    # 1. Ana sayfa testi
    try:
        response = requests.get(f"{base_url}/")
        print(f" Ana sayfa: {response.status_code}")
    except Exception as e:
        print(f" Ana sayfa hatası: {e}")

    # 2. Hakkında sayfası testi
    try:
        response = requests.get(f"{base_url}/about")
        print(f" Hakkında sayfası: {response.status_code}")
    except Exception as e:
        print(f" Hakkında sayfası hatası: {e}")

    # 3. Semptom listesi testi
    try:
        response = requests.get(f"{base_url}/symptoms")
        if response.status_code == 200:
            symptoms = response.json()
            print(f" Semptom listesi: {len(symptoms)} semptom")
        else:
            print(f" Semptom listesi hatası: {response.status_code}")
    except Exception as e:
        print(f" Semptom listesi hatası: {e}")

    # 4. Model listesi testi
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            models = response.json()
            print(f" Model listesi: {len(models)} model")
            print(f"   Modeller: {', '.join(models)}")
        else:
            print(f" Model listesi hatası: {response.status_code}")
    except Exception as e:
        print(f" Model listesi hatası: {e}")

    # 5. Tahmin testi
    test_cases = [
        {
            "name": "Ateş ve baş ağrısı",
            "symptoms": ["high_fever", "headache", "fatigue"],
            "model": "Random Forest"
        },
        {
            "name": "Mide problemi",
            "symptoms": ["stomach_pain", "nausea", "vomiting"],
            "model": "SVM"
        },
        {
            "name": "Solunum problemi",
            "symptoms": ["cough", "breathlessness", "chest_pain"],
            "model": "Logistic Regression"
        }
    ]

    print("\n Tahmin Testleri:")
    print("-" * 30)

    for i, test_case in enumerate(test_cases, 1):
        try:
            payload = {
                "symptoms": test_case["symptoms"],
                "model": test_case["model"]
            }

            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                print(f" Test {i} ({test_case['name']}):")
                print(f"   Semptomlar: {', '.join(test_case['symptoms'])}")
                print(f"   Model: {test_case['model']}")
                print(f"   Tahmin: {result.get('prediction', 'Bilinmiyor')}")

                if result.get('probabilities'):
                    print("   En yüksek olasılıklar:")
                    for disease, prob in result['probabilities'][:3]:
                        print(f"     - {disease}: {prob*100:.1f}%")
                print()
            else:
                print(f" Test {i} hatası: {response.status_code}")

        except Exception as e:
            print(f" Test {i} hatası: {e}")

    # 6. Hatalı istek testi
    print(" Hata Testleri:")
    print("-" * 20)

    try:
        # Boş semptom listesi
        response = requests.post(
            f"{base_url}/predict",
            json={"symptoms": [], "model": "Random Forest"},
            headers={"Content-Type": "application/json"}
        )
        print(f" Boş semptom testi: {response.status_code} (beklenen: 400)")

        # Geçersiz model
        response = requests.post(
            f"{base_url}/predict",
            json={"symptoms": ["fever"], "model": "NonExistentModel"},
            headers={"Content-Type": "application/json"}
        )
        print(f" Geçersiz model testi: {response.status_code}")

    except Exception as e:
        print(f" Hata testi hatası: {e}")

    print("\n" + "=" * 50)
    print(" API Testi Tamamlandı!")

if __name__ == "__main__":
    test_api()
