"""
Jupyter Notebook'ta eğitilen modelleri web uygulaması için dışa aktaran script
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

def load_and_prepare_data():
    """Veriyi yükle ve hazırla"""
    print("Veri yükleniyor...")

    # CSV dosyalarını yükle
    df_train = pd.read_csv('Training.csv')
    df_test = pd.read_csv('Testing.csv')

    print(f"Eğitim veri boyutu: {df_train.shape}")
    print(f"Test veri boyutu: {df_test.shape}")

    # Unnamed sütununu kaldır (varsa)
    if 'Unnamed: 133' in df_train.columns:
        df_train = df_train.drop('Unnamed: 133', axis=1)

    # Özellikleri ve hedef değişkeni ayır
    X_train = df_train.drop('prognosis', axis=1)
    y_train = df_train['prognosis']
    X_test = df_test.drop('prognosis', axis=1)
    y_test = df_test['prognosis']

    return X_train, y_train, X_test, y_test

def train_and_save_models():
    """Modelleri eğit ve kaydet"""
    print("Veri hazırlanıyor...")
    X_train, y_train, X_test, y_test = load_and_prepare_data()

    # Label encoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Modelleri tanımla
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', num_class=len(le.classes_), random_state=42),
        'LightGBM': lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), random_state=42, verbose=-1)
    }

    print("Modeller eğitiliyor ve kaydediliyor...")

    # Her modeli eğit ve kaydet
    for model_name, model in models.items():
        print(f"  - {model_name} eğitiliyor...")

        try:
            if model_name in ['XGBoost', 'LightGBM']:
                model.fit(X_train, y_train_encoded)
            else:
                model.fit(X_train, y_train)

            # Modeli kaydet
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"    ✓ {filename} kaydedildi")

            # Test doğruluğunu hesapla
            if model_name in ['XGBoost', 'LightGBM']:
                accuracy = model.score(X_test, y_test_encoded)
            else:
                accuracy = model.score(X_test, y_test)
            print(f"    Test doğruluğu: {accuracy:.4f}")

        except Exception as e:
            print(f"    ✗ Hata: {e}")

    # Label encoder'ı kaydet
    joblib.dump(le, 'label_encoder.pkl')
    print("✓ Label encoder kaydedildi")

    # Özellik adlarını kaydet
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, 'feature_names.pkl')
    print(f"✓ {len(feature_names)} özellik adı kaydedildi")

    # Hastalık isimlerini kaydet
    diseases = sorted(y_train.unique().tolist())
    joblib.dump(diseases, 'diseases.pkl')
    print(f"✓ {len(diseases)} hastalık adı kaydedildi")

    # Özet bilgi dosyası oluştur
    summary = {
        'total_models': len(models),
        'feature_count': len(feature_names),
        'disease_count': len(diseases),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': feature_names[:10],  # İlk 10 özellik
        'diseases': diseases[:10]  # İlk 10 hastalık
    }

    import json
    with open('model_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("✓ Model özeti kaydedildi")

    print("\n" + "="*50)
    print("TÜM MODELLER BAŞARIYLA OLUŞTURULDU!")
    print("="*50)
    print(f"Toplam model sayısı: {len(models)}")
    print(f"Özellik sayısı: {len(feature_names)}")
    print(f"Hastalık sayısı: {len(diseases)}")
    print("\nWeb uygulamasını başlatmak için: python app.py")

def create_sample_data():
    """Eğer CSV dosyaları yoksa örnek veri oluştur"""
    if not os.path.exists('Training.csv'):
        print("Training.csv bulunamadı. Örnek veri oluşturuluyor...")

        # Örnek semptomlar
        symptoms = [
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
            'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
            'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
            'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
            'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
            'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
            'indigestion', 'headache', 'yellowish_skin', 'dark_urine',
            'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
            'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever'
        ]

        # Örnek hastalıklar
        diseases = [
            'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
            'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes',
            'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
            'Migraine', 'Cervical spondylosis', 'Jaundice', 'Malaria',
            'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
            'Common Cold', 'Pneumonia', 'Heart attack', 'Hypothyroidism'
        ]

        # Rastgele veri oluştur
        np.random.seed(42)
        n_samples = 1000

        data = []
        for _ in range(n_samples):
            row = {}
            # Her semptom için rastgele 0 veya 1
            for symptom in symptoms:
                row[symptom] = np.random.choice([0, 1], p=[0.8, 0.2])  # %20 olasılıkla semptom var

            # Rastgele hastalık ata
            row['prognosis'] = np.random.choice(diseases)
            data.append(row)

        # DataFrame oluştur ve kaydet
        df_train = pd.DataFrame(data)
        df_train.to_csv('Training.csv', index=False)

        # Test verisi (daha küçük)
        test_data = []
        for _ in range(50):
            row = {}
            for symptom in symptoms:
                row[symptom] = np.random.choice([0, 1], p=[0.8, 0.2])
            row['prognosis'] = np.random.choice(diseases)
            test_data.append(row)

        df_test = pd.DataFrame(test_data)
        df_test.to_csv('Testing.csv', index=False)

        print("✓ Örnek veri dosyaları oluşturuldu")

if __name__ == "__main__":
    print("AI Hastalık Teşhis Sistemi - Model Oluşturucu")
    print("=" * 50)

    # Eğer CSV dosyaları yoksa örnek veri oluştur
    create_sample_data()

    # Modelleri eğit ve kaydet
    train_and_save_models()
