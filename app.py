from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)

# Global değişkenler
models = {}
feature_names = []
diseases = []
model_performances = {}
label_encoder = None

# Türkçe çeviri sözlükleri
disease_translations = {
    'Fungal infection': 'Mantar Enfeksiyonu',
    'Allergy': 'Alerji',
    'GERD': 'Reflü',
    'Chronic cholestasis': 'Kronik Kolestaz',
    'Drug Reaction': 'İlaç Reaksiyonu',
    'Peptic ulcer diseae': 'Peptik Ülser Hastalığı',
    'AIDS': 'AIDS',
    'Diabetes': 'Diyabet',
    'Gastroenteritis': 'Gastroenterit',
    'Bronchial Asthma': 'Bronşiyal Astım',
    'Hypertension': 'Hipertansiyon',
    'Migraine': 'Migren',
    'Cervical spondylosis': 'Servikal Spondiloz',
    'Paralysis (brain hemorrhage)': 'Felç (Beyin Kanaması)',
    'Jaundice': 'Sarılık',
    'Malaria': 'Sıtma',
    'Chicken pox': 'Su Çiçeği',
    'Dengue': 'Dang Virüsü',
    'Typhoid': 'Tifo',
    'hepatitis A': 'Hepatit A',
    'Hepatitis B': 'Hepatit B',
    'Hepatitis C': 'Hepatit C',
    'Hepatitis D': 'Hepatit D',
    'Hepatitis E': 'Hepatit E',
    'Alcoholic hepatitis': 'Alkolik Hepatit',
    'Tuberculosis': 'Tüberküloz',
    'Common Cold': 'Soğuk Algınlığı',
    'Pneumonia': 'Zatürre',
    'Dimorphic hemmorhoids(piles)': 'Hemoroid',
    'Heart attack': 'Kalp Krizi',
    'Varicose veins': 'Varis',
    'Hypothyroidism': 'Hipotiroidizm',
    'Hyperthyroidism': 'Hipertiroidizm',
    'Hypoglycemia': 'Hipoglisemi',
    'Osteoarthristis': 'Osteoartrit',
    'Arthritis': 'Artrit',
    '(vertigo) Paroymsal  Positional Vertigo': 'Vertigo',
    'Acne': 'Akne',
    'Urinary tract infection': 'İdrar Yolu Enfeksiyonu',
    'Psoriasis': 'Sedef Hastalığı',
    'Impetigo': 'İmpetigo'
}

symptom_translations = {
    'itching': 'Kaşıntı',
    'skin_rash': 'Cilt Döküntüsü',
    'nodal_skin_eruptions': 'Nodüler Cilt Döküntüleri',
    'continuous_sneezing': 'Sürekli Hapşırma',
    'shivering': 'Titreme',
    'chills': 'Üşüme',
    'joint_pain': 'Eklem Ağrısı',
    'stomach_pain': 'Mide Ağrısı',
    'acidity': 'Midede Ekşime',
    'ulcers_on_tongue': 'Dil Üzerinde Ülser',
    'muscle_wasting': 'Kas Erimesi',
    'vomiting': 'Kusma',
    'burning_micturition': 'İdrar Yaparken Yanma',
    'spotting_ urination': 'Kanla Karışık İdrar',
    'fatigue': 'Yorgunluk',
    'weight_gain': 'Kilo Alma',
    'anxiety': 'Anksiyete',
    'cold_hands_and_feets': 'Soğuk El ve Ayaklar',
    'mood_swings': 'Ruh Hali Değişiklikleri',
    'weight_loss': 'Kilo Kaybı',
    'restlessness': 'Huzursuzluk',
    'lethargy': 'Uyuşukluk',
    'patches_in_throat': 'Boğazda Lekeler',
    'irregular_sugar_level': 'Düzensiz Şeker Seviyesi',
    'cough': 'Öksürük',
    'high_fever': 'Yüksek Ateş',
    'sunken_eyes': 'Çökük Gözler',
    'breathlessness': 'Nefes Darlığı',
    'sweating': 'Terleme',
    'dehydration': 'Dehidrasyon',
    'indigestion': 'Hazımsızlık',
    'headache': 'Baş Ağrısı',
    'yellowish_skin': 'Sarımsı Cilt',
    'dark_urine': 'Koyu İdrar',
    'nausea': 'Mide Bulantısı',
    'loss_of_appetite': 'İştahsızlık',
    'pain_behind_the_eyes': 'Göz Arkası Ağrısı',
    'back_pain': 'Sırt Ağrısı',
    'constipation': 'Kabızlık',
    'abdominal_pain': 'Karın Ağrısı',    'diarrhoea': 'İshal',
    'mild_fever': 'Hafif Ateş',
    'yellow_urine': 'Sarı İdrar',
    # Ek semptom çevirileri
    'yellowing_of_eyes': 'Gözlerde Sarılaşma',
    'acute_liver_failure': 'Akut Karaciğer Yetmezliği',
    'fluid_overload': 'Hipervolemi',
    'swelling_of_stomach': 'Mide Şişmesi',
    'swelled_lymph_nodes': 'Şişmiş Lenf Düğümleri',
    'malaise': 'Halsizlik',
    'blurred_and_distorted_vision': 'Bulanık ve Bozuk Görüş',
    'phlegm': 'Balgam',
    'throat_irritation': 'Boğaz Tahrişi',
    'redness_of_eyes': 'Göz Kızarıklığı',
    'sinus_pressure': 'Sinüs Basıncı',
    'runny_nose': 'Burun Akması',
    'congestion': 'Tıkanıklık',
    'chest_pain': 'Göğüs Ağrısı',
    'weakness_in_limbs': 'Uzuvlarda Güçsüzlük',
    'fast_heart_rate': 'Hızlı Kalp Atışı',
    'pain_during_bowel_movements': 'Bağırsak Hareketlerinde Ağrı',
    'pain_in_anal_region': 'Kuyruk Bölgesinde Ağrı',
    'bloody_stool': 'Kanlı Dışkı',
    'irritation_in_anus': 'Pişik',
    'neck_pain': 'Boyun Ağrısı',
    'dizziness': 'Baş Dönmesi',
    'cramps': 'Kramplar',
    'bruising': 'Morluk',
    'obesity': 'Obezite',
    'swollen_legs': 'Şişmiş Bacaklar',
    'swollen_blood_vessels': 'Şişmiş Kan Damarları',
    'puffy_face_and_eyes': 'Şişmiş Yüz ve Gözler',
    'enlarged_thyroid': 'Büyümüş Tiroid',
    'brittle_nails': 'Kırılgan Tırnaklar',
    'swollen_extremeties': 'Şişmiş Uzuvlar',
    'excessive_hunger': 'Aşırı Açlık',
    'extra_marital_contacts': 'Evlilik Dışı İlişkiler',
    'drying_and_tingling_lips': 'Kuruyup Karıncalanan Dudaklar',
    'slurred_speech': 'Peltekleşen Konuşma',
    'knee_pain': 'Diz Ağrısı',
    'hip_joint_pain': 'Kalça Eklem Ağrısı',
    'muscle_weakness': 'Kas Güçsüzlüğü',
    'stiff_neck': 'Sert Boyun',
    'swelling_joints': 'Şişen Eklemler',
    'movement_stiffness': 'Sallanma Hissi',
    'spinning_movements': 'Sersemlik',
    'loss_of_balance': 'Denge Kaybı',
    'unsteadiness': 'Dengesizlik',
    'weakness_of_one_body_side': 'Vücudun Bir Tarafında Güçsüzlük',
    'loss_of_smell': 'Koku Alma Kaybı',
    'bladder_discomfort': 'Mesane Rahatsızlığı',
    'foul_smell_of urine': 'İdrarda Kötü Koku',
    'continuous_feel_of_urine': 'Sürekli İdrar Hissi',
    'passage_of_gases': 'Gaz Sancısı',
    'internal_itching': 'İç Kaşıntı',
    'toxic_look_(typhos)': 'Toksik Görünüm (Tifo)',
    'depression': 'Depresyon',
    'irritability': 'Sinirlilik',
    'muscle_pain': 'Kas Ağrısı',
    'altered_sensorium': 'Kafa Karışıklığı',
    'red_spots_over_body': 'Vücutta Kırmızı Lekeler',
    'belly_pain': 'Karın Ağrısı',
    'abnormal_menstruation': 'Anormal Adet',
    'dischromic _patches': 'Ciltte Renk Değişiklikleri',
    'watering_from_eyes': 'Gözlerden Sulanma',
    'increased_appetite': 'Artan İştah',
    'polyuria': 'Aşırı İdrar İhtiyacı',
    'family_history': 'Aile Geçmişi',
    'mucoid_sputum': 'Mukozlu Balgam',
    'rusty_sputum': 'Paslı Balgam',
    'lack_of_concentration': 'Konsantrasyon Eksikliği',
    'visual_disturbances': 'Görme Bozuklukları',
    'receiving_blood_transfusion': 'Kan Transfüzyonu Almak',
    'receiving_unsterile_injections': 'Steril Olmayan Enjeksiyon Almak',
    'coma': 'Koma',
    'stomach_bleeding': 'Mide Kanaması',
    'distention_of_abdomen': 'Karında Şişkinlik',
    'history_of_alcohol_consumption': 'Alkol Kullanım Geçmişi',
    'fluid_overload.1': 'Sıvı Yüklenmesi',
    'blood_in_sputum': 'Balgamda Kan',
    'prominent_veins_on_calf': 'Baldırda Belirgin Damarlar',
    'palpitations': 'Çarpıntı',
    'painful_walking': 'Yürürken Ağrı',
    'pus_filled_pimples': 'İrinli Sivilceler',
    'blackheads': 'Siyah Noktalar',
    'scurring': 'Kabuk Oluşumu',
    'skin_peeling': 'Cilde Soyulma',
    'silver_like_dusting': 'Ciltte Gümüş Benzeri Pullanma',
    'small_dents_in_nails': 'Tırnaklarda Küçük Çukurlar',
    'inflammatory_nails': 'İltihaplı Tırnaklar',
    'blister': 'Su Toplama',
    'red_sore_around_nose': 'Burun Çevresinde Kırmızı Yara',
    'yellow_crust_ooze': 'Sarı Kabuklu Akıntı',
}

def get_turkish_disease_name(english_name):
    """İngilizce hastalık ismini Türkçeye çevir"""
    return disease_translations.get(english_name, english_name)

def get_turkish_symptom_name(english_name):
    """İngilizce semptom ismini Türkçeye çevir"""
    return symptom_translations.get(english_name, english_name.replace('_', ' ').title())

def load_models_and_data():
    """Modelleri ve veri özelliklerini yükle"""
    global models, feature_names, diseases, model_performances, label_encoder

    try:        # Eğer model dosyaları mevcutsa yükle
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'SVM': 'svm_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'KNN': 'knn_model.pkl',
            'Gradient Boosting': 'gradient_boosting_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
            'LightGBM': 'lightgbm_model.pkl'
        }

        for name, filename in model_files.items():
            if os.path.exists(filename):
                models[name] = joblib.load(filename)

        # Label encoder'ı yükle (XGBoost ve LightGBM için gerekli)
        if os.path.exists('label_encoder.pkl'):
            label_encoder = joblib.load('label_encoder.pkl')

        # Özellik adlarını yükle
        if os.path.exists('feature_names.pkl'):
            feature_names = joblib.load('feature_names.pkl')
        else:
            # CSV'den özellik adlarını al
            df = pd.read_csv('Training.csv')
            feature_names = [col for col in df.columns if col != 'prognosis']

        # Hastalık isimlerini yükle
        if os.path.exists('diseases.pkl'):
            diseases = joblib.load('diseases.pkl')
        else:
            df = pd.read_csv('Training.csv')
            diseases = sorted(df['prognosis'].unique().tolist())

        print(f"Yüklenen modeller: {list(models.keys())}")
        print(f"Toplam özellik sayısı: {len(feature_names)}")
        print(f"Toplam hastalık sayısı: {len(diseases)}")
        print(f"Label encoder yüklendi: {label_encoder is not None}")

    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        # Fallback veriler
        feature_names = get_default_symptoms()
        diseases = get_default_diseases()

def get_default_symptoms():
    """Varsayılan semptom listesi"""
    return [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
        'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
        'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
        'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
        'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
        'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
        'dehydration', 'indigestion', 'headache', 'yellowish_skin',
        'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
        'mild_fever', 'yellow_urine'
    ]

def get_default_diseases():
    """Varsayılan hastalık listesi"""
    return [
        'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
        'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes',
        'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
        'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
        'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid',
        'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
        'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
        'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
        'Heart attack', 'Varicose veins', 'Hypothyroidism',
        'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
        'Arthritis', '(vertigo) Paroymsal  Positional Vertigo',
        'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'
    ]

def predict_disease(symptoms_input, model_name='Random Forest'):
    """Semptomları kullanarak hastalık tahmini yap"""
    try:
        # Semptom vektörü oluştur
        symptoms_vector = np.zeros(len(feature_names))

        for symptom in symptoms_input:
            if symptom in feature_names:
                idx = feature_names.index(symptom)
                symptoms_vector[idx] = 1        # Model tahmini
        if model_name in models:
            model = models[model_name]
            raw_prediction = model.predict([symptoms_vector])[0]


            if hasattr(raw_prediction, 'item'):
                prediction = raw_prediction.item()
            else:
                prediction = raw_prediction
              # Eğer prediction string değilse, diseases listesinden karşılığını al
            if isinstance(prediction, (int, float, np.integer, np.floating)):
                prediction_idx = int(prediction)

                # XGBoost ve LightGBM için label encoder kullan
                if model_name in ['XGBoost', 'LightGBM'] and label_encoder is not None:
                    if prediction_idx < len(label_encoder.classes_):
                        prediction = str(label_encoder.classes_[prediction_idx])
                    else:
                        prediction = str(prediction)
                # Diğer modeller için normal diseases listesi kullan
                elif hasattr(model, 'classes_') and prediction_idx < len(model.classes_):
                    prediction = str(model.classes_[prediction_idx])
                elif prediction_idx < len(diseases):
                    prediction = str(diseases[prediction_idx])
                else:
                    prediction = str(prediction)
            else:
                prediction = str(prediction)# Eğer model olasılık verebilirse
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([symptoms_vector])[0]
                # Olasılıkları da Python floata dönüştür
                probabilities = [float(prob) for prob in probabilities]
                  # Doğru hastalık listesini kullan
                if model_name in ['XGBoost', 'LightGBM'] and label_encoder is not None:
                    # XGBoost ve LightGBM için label encoder kullan
                    disease_list = [str(disease) for disease in label_encoder.classes_[:len(probabilities)]]
                elif hasattr(model, 'classes_') and len(model.classes_) == len(probabilities):
                    disease_list = [str(disease) for disease in model.classes_]
                else:
                    disease_list = diseases[:len(probabilities)]

                disease_probs = list(zip(disease_list, probabilities))
                disease_probs.sort(key=lambda x: x[1], reverse=True)
                turkish_probs = [(get_turkish_disease_name(disease), float(prob)) for disease, prob in disease_probs[:5]]

                # Detaylı hastalık bilgisini ekle
                turkish_disease_name = get_turkish_disease_name(str(prediction))
                disease_details = get_disease_info(turkish_disease_name)

                return {
                    'prediction': turkish_disease_name,
                    'prediction_english': str(prediction),
                    'probabilities': turkish_probs,
                    'model_used': model_name,
                    'disease_info': disease_details
                }
            else:
                # Detaylı hastalık bilgisini ekle
                turkish_disease_name = get_turkish_disease_name(str(prediction))
                disease_details = get_disease_info(turkish_disease_name)

                return {
                    'prediction': turkish_disease_name,
                    'prediction_english': str(prediction),
                    'probabilities': [],
                    'model_used': model_name,                    'disease_info': disease_details
                }
        else:
            # Basit kural tabanlı tahmin (fallback)
            fallback_disease = get_turkish_disease_name('Common Cold')
            disease_details = get_disease_info(fallback_disease)

            return {
                'prediction': fallback_disease,
                'prediction_english': 'Common Cold',
                'probabilities': [(fallback_disease, 0.7)],
                'model_used': 'Rule-based (fallback)',
                'disease_info': disease_details
            }

    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return {
            'prediction': 'Teşhis yapılamadı',
            'probabilities': [],
            'model_used': 'Error',
            'error': str(e)
        }

@app.route('/')
def index():
    """Ana sayfa"""
    # Semptomları Türkçe çeviri ile birlikte hazırla
    turkish_symptoms = []
    for symptom in feature_names:
        turkish_symptoms.append({
            'english': symptom,
            'turkish': get_turkish_symptom_name(symptom)
        })

    return render_template('index.html',
                         symptoms=turkish_symptoms,
                         models=list(models.keys()) if models else ['Random Forest'])

@app.route('/predict', methods=['POST'])
def predict():
    """Hastalık tahmini endpoint'i"""
    try:
        data = request.json
        selected_symptoms = data.get('symptoms', [])
        model_name = data.get('model', 'Random Forest')

        if not selected_symptoms:
            return jsonify({'error': 'En az bir semptom seçmelisiniz'}), 400

        result = predict_disease(selected_symptoms, model_name)

        # Sonuçları logla
        log_prediction(selected_symptoms, result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500

@app.route('/symptoms')
def get_symptoms():
    """Semptom listesini döndür"""
    return jsonify(feature_names)

@app.route('/symptoms-turkish')
def get_symptoms_turkish():
    """Semptom listesini Türkçe olarak döndür"""
    turkish_symptoms = []
    for symptom in feature_names:
        turkish_symptoms.append({
            'english': symptom,
            'turkish': get_turkish_symptom_name(symptom)
        })
    return jsonify(turkish_symptoms)

@app.route('/models')
def get_models():
    """Mevcut model listesini döndür"""
    return jsonify(list(models.keys()) if models else ['Random Forest'])

@app.route('/about')
def about():
    """Hakkında sayfası"""
    return render_template('about.html',
                         model_count=len(models),
                         symptom_count=len(feature_names),
                         disease_count=len(diseases))

def log_prediction(symptoms, result):
    """Tahmin sonuçlarını logla"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symptoms': symptoms,
            'prediction': result.get('prediction'),
            'model_used': result.get('model_used'),
            'probabilities': result.get('probabilities', [])
        }

        # Log dosyasına yaz
        with open('predictions.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Log yazma hatası: {e}")

# HASTALIK AÇIKLAMALARI
disease_info = {
    'Mantar Enfeksiyonu': {
        'description': 'Vücudun çeşitli bölgelerinde mantarların neden olduğu enfeksiyon türüdür. Genellikle nem ve sıcaklığın yoğun olduğu bölgelerde görülür.',
        'symptoms': ['Kaşıntı', 'Cilt döküntüsü', 'Kızarıklık', 'Yanma hissi'],
        'causes': ['Nem ve sıcaklık', 'Zayıf bağışıklık sistemi', 'Hijyen eksikliği', 'Dar ve hava almayan giysiler'],
        'treatment': ['Antifungal kremler', 'Hijyen kurallarına dikkat', 'Kuru ve temiz tutma', 'Uygun giyim'],
        'prevention': ['Günlük banyo', 'Temiz ve kuru giysi', 'Hava alan ayakkabı', 'Nemli ortamlardan kaçınma'],
        'when_to_see_doctor': ['Semptomlar 2 haftadan fazla sürerse', 'Enfeksiyon yayılırsa', 'Ateş çıkarsa', 'Şiddetli ağrı varsa'],
        'severity': 'Hafif-Orta'
    },
    'Alerji': {
        'description': 'Bağışıklık sisteminin normalde zararsız olan maddelere karşı aşırı tepki vermesidir.',
        'symptoms': ['Hapşırma', 'Burun akması', 'Kaşıntı', 'Göz sulanması', 'Cilt döküntüleri'],
        'causes': ['Polen', 'Ev tozu akarları', 'Hayvan tüyleri', 'Belirli gıdalar', 'İlaçlar'],
        'treatment': ['Antihistaminik ilaçlar', 'Allerjen kaçınma', 'Nazal spreyler', 'Bağışıklık tedavisi'],
        'prevention': ['Allerjen tespiti', 'Ev temizliği', 'Hava filtreleri', 'Beslenme dikkat'],
        'when_to_see_doctor': ['Nefes darlığı', 'Şiddetli reaksiyon', 'Günlük yaşamı etkileme', 'İlaç etkisizliği'],
        'severity': 'Hafif-Ciddi'
    },
    'Reflü': {
        'description': 'Mide asidinin yemek borusuna geri kaçması sonucu oluşan rahatsızlıktır.',
        'symptoms': ['Göğüs yanması', 'Ekşi tatma', 'Yutma güçlüğü', 'Öksürük', 'Ses kısıklığı'],
        'causes': ['Beslenme alışkanlıkları', 'Obezite', 'Hamilelik', 'Sigara', 'Stres'],
        'treatment': ['Beslenme değişikliği', 'Kilo verme', 'Asit azaltıcı ilaçlar', 'Yaşam tarzı değişikliği'],
        'prevention': ['Küçük porsiyonlar', 'Yatmadan 3 saat önce yememe', 'Asitli gıdalardan kaçınma', 'Başı yüksek uyuma'],
        'when_to_see_doctor': ['Sürekli semptomlar', 'Yutma güçlüğü', 'Kilo kaybı', 'Kan kusma'],
        'severity': 'Hafif-Orta'
    },
    'Diyabet': {
        'description': 'Kan şeker seviyesinin sürekli yüksek olduğu kronik hastalıktır.',
        'symptoms': ['Aşırı susama', 'Sık idrara çıkma', 'Yorgunluk', 'Bulanık görme', 'Yavaş iyileşen yaralar'],
        'causes': ['Genetik faktörler', 'Obezite', 'Sedanter yaşam', 'Yaş', 'Hormon değişiklikleri'],
        'treatment': ['İnsülin tedavisi', 'Kan şekeri takibi', 'Diyet kontrolü', 'Düzenli egzersiz', 'İlaç tedavisi'],
        'prevention': ['Sağlıklı beslenme', 'Düzenli egzersiz', 'Kilo kontrolü', 'Düzenli kontroller'],
        'when_to_see_doctor': ['Kan şekeri 250 mg/dl üstü', 'Bilinç bulanıklığı', 'Şiddetli dehidrasyon', 'Acil durumlar'],
        'severity': 'Ciddi'
    },
    'Hipertansiyon': {
        'description': 'Kan basıncının sürekli olarak normal değerlerin üstünde olmasıdır.',
        'symptoms': ['Baş ağrısı', 'Baş dönmesi', 'Göğüs ağrısı', 'Nefes darlığı', 'Yorgunluk'],
        'causes': ['Genetik', 'Yaş', 'Obezite', 'Tuz tüketimi', 'Stres', 'Sigara'],
        'treatment': ['Antihipertansif ilaçlar', 'Beslenme düzenleme', 'Egzersiz', 'Stres yönetimi'],
        'prevention': ['Az tuz', 'Düzenli spor', 'Kilo kontrolü', 'Sigara bırakma', 'Stres azaltma'],
        'when_to_see_doctor': ['Tansiyon 180/120 üstü', 'Şiddetli baş ağrısı', 'Görme bozukluğu', 'Göğüs ağrısı'],
        'severity': 'Ciddi'
    },
    'Migren': {
        'description': 'Şiddetli, zonklayıcı baş ağrısı atakları ile karakterize nörolojik hastalıktır.',
        'symptoms': ['Şiddetli baş ağrısı', 'Bulantı', 'Işık hassasiyeti', 'Ses hassasiyeti', 'Görme bozuklukları'],
        'causes': ['Genetik', 'Hormon değişiklikleri', 'Stres', 'Beslenme', 'Uyku düzensizliği'],
        'treatment': ['Ağrı kesici ilaçlar', 'Preventif ilaçlar', 'Yaşam tarzı değişiklikleri', 'Stres yönetimi'],
        'prevention': ['Düzenli uyku', 'Stres azaltma', 'Tetikleyici faktörlerden kaçınma', 'Düzenli beslenme'],
        'when_to_see_doctor': ['Ani şiddetli baş ağrısı', 'Ateş ile birlikte', 'Bilinç değişikliği', 'Sık tekrarlama'],
        'severity': 'Orta-Ciddi'
    },
    'Soğuk Algınlığı': {
        'description': 'Üst solunum yollarının viral enfeksiyonuudur. En yaygın görülen hastalık türlerinden biridir.',
        'symptoms': ['Burun akması', 'Hapşırma', 'Öksürük', 'Boğaz ağrısı', 'Hafif ateş'],
        'causes': ['Rinovirüs', 'Soğuk hava', 'Düşük bağışıklık', 'Yakın temas', 'Stres'],
        'treatment': ['İstirahat', 'Bol sıvı', 'Ağrı kesici', 'Burun damlası', 'Gargara'],
        'prevention': ['El yıkama', 'Sosyal mesafe', 'Bağışıklığı güçlendirme', 'Vitamin C', 'Sağlıklı beslenme'],
        'when_to_see_doctor': ['10 günden fazla sürerse', 'Yüksek ateş', 'Şiddetli baş ağrısı', 'Nefes darlığı'],
        'severity': 'Hafif'
    },
    'Zatürre': {
        'description': 'Akciğerlerin enfeksiyonu sonucu oluşan ciddi hastalıktır.',
        'symptoms': ['Yüksek ateş', 'Öksürük', 'Nefes darlığı', 'Göğüs ağrısı', 'Balgam'],
        'causes': ['Bakteriler', 'Virüsler', 'Mantarlar', 'Düşük bağışıklık', 'Yaş'],
        'treatment': ['Antibiyotik', 'Hastane yatışı', 'Oksijen desteği', 'Sıvı tedavisi', 'İstirahat'],
        'prevention': ['Aşılama', 'El hijyeni', 'Sigara bırakma', 'Bağışıklığı güçlendirme'],
        'when_to_see_doctor': ['Nefes darlığı', 'Yüksek ateş', 'Göğüs ağrısı', 'Bilinç bulanıklığı'],
        'severity': 'Ciddi'
    },    'Kalp Krizi': {
        'description': 'Kalp kasının bir bölümünün kan akışı kesintisi nedeniyle zarar görmesidir.',
        'symptoms': ['Şiddetli göğüs ağrısı', 'Nefes darlığı', 'Terleme', 'Bulantı', 'Sol kola yayılan ağrı'],
        'causes': ['Koroner arter hastalığı', 'Yüksek kolesterol', 'Hipertansiyon', 'Diyabet', 'Sigara'],
        'treatment': ['ACİL TIBBİ MÜDAHALE', 'Anjiyoplasti', 'İlaç tedavisi', 'Bypass', 'Rehabilitasyon'],
        'prevention': ['Sağlıklı beslenme', 'Düzenli egzersiz', 'Sigara bırakma', 'Stres yönetimi', 'Düzenli kontrol'],
        'when_to_see_doctor': ['HEMEN 112 ARAYIN', 'Göğüs ağrısı', 'Nefes darlığı', 'Baygınlık'],
        'severity': 'Çok Ciddi - ACİL'
    },
    'Kronik Kolestaz': {
        'description': 'Safra akışının azalması veya durması sonucu oluşan karaciğer hastalığıdır.',
        'symptoms': ['Kaşıntı', 'Sarılık', 'Koyu idrar', 'Açık renkli dışkı', 'Yorgunluk'],
        'causes': ['Safra taşları', 'Karaciğer hastalıkları', 'İlaçlar', 'Enfeksiyonlar', 'Tümörler'],
        'treatment': ['Altta yatan nedeni tedavi', 'Kaşıntı giderici ilaçlar', 'Vitamin takviyeleri', 'Safra asidi bağlayıcıları'],
        'prevention': ['Sağlıklı beslenme', 'Alkol kısıtlaması', 'Düzenli kontroller', 'İlaç takibi'],
        'when_to_see_doctor': ['Kalıcı sarılık', 'Şiddetli kaşıntı', 'Karın ağrısı', 'Ateş'],
        'severity': 'Orta-Ciddi'
    },
    'İlaç Reaksiyonu': {
        'description': 'İlaçlara karşı vücudun gösterdiği istenmeyen yan etkilerdir.',
        'symptoms': ['Cilt döküntüsü', 'Kaşıntı', 'Nefes darlığı', 'Şişme', 'Bulantı'],
        'causes': ['Antibiyotikler', 'Ağrı kesiciler', 'Duyarlılık', 'Doz aşımı', 'İlaç etkileşimi'],
        'treatment': ['İlacı durdurma', 'Antihistaminikler', 'Kortikosteroidler', 'Adrenalin (ciddi durumlarda)'],
        'prevention': ['İlaç alerjisi testi', 'Doktor danışmanlığı', 'Geçmiş reaksiyon bildirimi', 'Dikkatli ilaç kullanımı'],
        'when_to_see_doctor': ['Nefes darlığı', 'Şiddetli döküntü', 'Yüz şişmesi', 'Bilinç değişikliği'],
        'severity': 'Hafif-Çok Ciddi'
    },
    'Peptik Ülser Hastalığı': {
        'description': 'Mide veya on iki parmak bağırsağının iç duvarında oluşan yaralar.',
        'symptoms': ['Mide ağrısı', 'Yanma hissi', 'Şişkinlik', 'Bulantı', 'Kan kusma'],
        'causes': ['H. pylori bakterisi', 'NSAİİ kullanımı', 'Stres', 'Sigara', 'Alkol'],
        'treatment': ['Antibiyotik tedavisi', 'Asit azaltıcı ilaçlar', 'Beslenme düzeni', 'Stres yönetimi'],
        'prevention': ['H. pylori eradikasyonu', 'NSAİİ dikkatli kullanım', 'Sigara bırakma', 'Düzenli beslenme'],
        'when_to_see_doctor': ['Şiddetli karın ağrısı', 'Kan kusma', 'Siyah dışkı', 'Ani kilo kaybı'],
        'severity': 'Orta-Ciddi'
    },
    'AIDS': {
        'description': 'HIV virüsünün neden olduğu kazanılmış bağışıklık yetmezliği sendromudur.',
        'symptoms': ['Sürekli yorgunluk', 'Beklenmedik kilo kaybı', 'Gece terlemesi', 'Tekrarlayan enfeksiyonlar'],
        'causes': ['HIV virüsü', 'Güvenli olmayan cinsel ilişki', 'Kontamine kan', 'Anne-bebek bulaşması'],
        'treatment': ['Antiretroviral tedavi', 'Fırsatçı enfeksiyon tedavisi', 'Bağışıklık desteği', 'Yaşam kalitesi iyileştirme'],
        'prevention': ['Güvenli cinsel ilişki', 'Steril enjektör kullanımı', 'Düzenli test', 'Eğitim'],
        'when_to_see_doctor': ['Risk grubunda olma', 'Şüpheli semptomlar', 'Düzenli takip', 'Partner pozitifse'],
        'severity': 'Çok Ciddi'
    },
    'Gastroenterit': {
        'description': 'Mide ve bağırsağın iltihaplanması sonucu oluşan hastalıktır.',
        'symptoms': ['İshal', 'Kusma', 'Karın ağrısı', 'Ateş', 'Dehidrasyon'],
        'causes': ['Viral enfeksiyonlar', 'Bakteriyel enfeksiyonlar', 'Bozuk gıda', 'Parazitler', 'Toksinler'],
        'treatment': ['Sıvı replasmanı', 'Probiyotikler', 'BRAT diyeti', 'Elektrolit dengesi', 'İstirahat'],
        'prevention': ['El hijyeni', 'Güvenli gıda tüketimi', 'Temiz su', 'Aşılama', 'Kişisel hijyen'],
        'when_to_see_doctor': ['Şiddetli dehidrasyon', 'Kanlı ishal', 'Yüksek ateş', '3 günden uzun süre'],
        'severity': 'Hafif-Orta'
    },
    'Bronşiyal Astım': {
        'description': 'Hava yollarının daralması ile karakterize kronik solunum hastalığıdır.',
        'symptoms': ['Nefes darlığı', 'Hırıltı', 'Öksürük', 'Göğüs sıkışması', 'Nefes alma güçlüğü'],
        'causes': ['Allerjenler', 'Hava kirliliği', 'Soğuk hava', 'Egzersiz', 'Stres'],
        'treatment': ['Bronkodilatör ilaçlar', 'İnhaler kortikosteroidler', 'Allerjen kaçınma', 'Acil durum planı'],
        'prevention': ['Tetikleyicilerden kaçınma', 'Düzenli ilaç kullanımı', 'İnhaler tekniği', 'Egzersiz programı'],
        'when_to_see_doctor': ['Şiddetli nefes darlığı', 'İlaçlara yanıtsızlık', 'Sürekli semptomlar', 'Acil durum'],
        'severity': 'Orta-Ciddi'
    },
    'Servikal Spondiloz': {
        'description': 'Boyun omurlarında yaşa bağlı dejeneratif değişikliklerdir.',
        'symptoms': ['Boyun ağrısı', 'Sertlik', 'Baş ağrısı', 'Kol uyuşması', 'Hareket kısıtlılığı'],
        'causes': ['Yaşlanma', 'Yıpranma', 'Kötü postür', 'Travma', 'Genetik faktörler'],
        'treatment': ['Fizik tedavi', 'Ağrı kesici ilaçlar', 'Kas gevşeticiler', 'Egzersiz', 'Boyunluk'],
        'prevention': ['Doğru postür', 'Düzenli egzersiz', 'Ergonomik çalışma', 'Stres azaltma'],
        'when_to_see_doctor': ['Şiddetli ağrı', 'Kol güçsüzlüğü', 'Uyuşukluk', 'Hareket kaybı'],
        'severity': 'Hafif-Orta'
    },
    'Felç (Beyin Kanaması)': {
        'description': 'Beyin damarlarının patlama veya tıkanması sonucu oluşan ciddi hastalıktır.',
        'symptoms': ['Ani konuşma bozukluğu', 'Yüz felci', 'Kol-bacak güçsüzlüğü', 'Şiddetli baş ağrısı', 'Bilinç kaybı'],
        'causes': ['Hipertansiyon', 'Arteriyoskleroz', 'Anevrizmal', 'Kan pıhtılaşma bozukluğu', 'Yaş'],
        'treatment': ['ACİL MÜDAHALE', 'Kan akışını restore etme', 'Fizik tedavi', 'Konuşma terapisi', 'Rehabilitasyon'],
        'prevention': ['Kan basıncı kontrolü', 'Kolesterol yönetimi', 'Düzenli egzersiz', 'Sigara bırakma'],
        'when_to_see_doctor': ['HEMEN 112 ARAYIN', 'Ani semptomlar', 'FAST testi pozitif', 'Acil durum'],
        'severity': 'Çok Ciddi - ACİL'
    },
    'Sarılık': {
        'description': 'Kan ve dokularda bilirubin artışı sonucu sarı renk değişimi.',
        'symptoms': ['Deri ve gözlerde sarılaşma', 'Koyu idrar', 'Açık dışkı', 'Kaşıntı', 'Yorgunluk'],
        'causes': ['Hepatit', 'Safra taşları', 'Karaciğer hastalıkları', 'Kan hastalıkları', 'İlaçlar'],
        'treatment': ['Altta yatan nedeni tedavi', 'Karaciğer desteği', 'Beslenme düzeni', 'İstirahat'],
        'prevention': ['Hepatit aşısı', 'Hijyen kuralları', 'Güvenli kan transfüzyonu', 'Alkol kısıtlaması'],
        'when_to_see_doctor': ['Ani sarılık', 'Karın ağrısı', 'Ateş', 'Bilinç değişikliği'],
        'severity': 'Hafif-Ciddi'
    },
    'Sıtma': {
        'description': 'Plasmodium parazitinin neden olduğu sivrisinek kaynaklı hastalık.',
        'symptoms': ['Yüksek ateş', 'Titreme', 'Terleme', 'Baş ağrısı', 'Kas ağrıları'],
        'causes': ['Anopheles sivrisinek ısırığı', 'Plasmodium paraziti', 'Tropik bölge seyahati'],
        'treatment': ['Antimalaryal ilaçlar', 'Ateş düşürücüler', 'Sıvı desteği', 'Hastane bakımı'],
        'prevention': ['Sivrisinek korunması', 'Profilaksi ilaçları', 'Cibinlik kullanımı', 'Repellent'],
        'when_to_see_doctor': ['Tropik seyahat sonrası ateş', 'Şiddetli semptomlar', 'Bilinç bulanıklığı'],
        'severity': 'Orta-Ciddi'
    },
    'Su Çiçeği': {
        'description': 'Varicella-zoster virüsünün neden olduğu bulaşıcı hastalık.',
        'symptoms': ['Kaşıntılı su dolu kabarcıklar', 'Ateş', 'Baş ağrısı', 'Yorgunluk', 'İştahsızlık'],
        'causes': ['Varicella-zoster virüsü', 'Damlacık yoluyla bulaşma', 'Direkt temas'],
        'treatment': ['Antiviral ilaçlar', 'Kaşıntı gidericiler', 'Ateş düşürücüler', 'İzolasyon'],
        'prevention': ['Aşılama', 'Hasta kişilerden uzak durma', 'Hijyen kuralları'],
        'when_to_see_doctor': ['Yüksek ateş', 'Nefes darlığı', 'Enfeksiyon belirtileri', 'Komplikasyonlar'],
        'severity': 'Hafif-Orta'
    },
    'Dang Virüsü': {
        'description': 'Aedes sivrisinekleri tarafından bulaştırılan viral hastalık.',
        'symptoms': ['Yüksek ateş', 'Şiddetli baş ağrısı', 'Göz arkası ağrısı', 'Kas ağrıları', 'Döküntü'],
        'causes': ['Dengue virüsü', 'Aedes aegypti sivrisinek', 'Tropik bölgeler'],
        'treatment': ['Semptomatik tedavi', 'Sıvı replasmanı', 'Ateş kontrolü', 'Kanama takibi'],
        'prevention': ['Sivrisinek kontrolü', 'Su birikintilerini temizleme', 'Koruyucu giysi'],
        'when_to_see_doctor': ['Şiddetli karın ağrısı', 'Kanama', 'Nefes darlığı', 'Düşük tansiyon'],
        'severity': 'Orta-Ciddi'
    },
    'Tifo': {
        'description': 'Salmonella typhi bakterisinin neden olduğu sistemik enfeksiyon.',
        'symptoms': ['Sürekli yüksek ateş', 'Baş ağrısı', 'Karın ağrısı', 'Döküntü', 'İshal'],
        'causes': ['Salmonella typhi', 'Kontamine su ve gıda', 'Kötü hijyen koşulları'],
        'treatment': ['Antibiyotik tedavisi', 'Sıvı elektrolit dengesi', 'İstirahat', 'İzolasyon'],
        'prevention': ['Aşılama', 'Güvenli su tüketimi', 'Hijyen kuralları', 'Temiz gıda'],
        'when_to_see_doctor': ['Sürekli yüksek ateş', 'Şiddetli karın ağrısı', 'Bilinç bulanıklığı'],
        'severity': 'Ciddi'
    },
    'Hepatit A': {
        'description': 'Hepatit A virüsünün neden olduğu akut karaciğer enfeksiyonu.',
        'symptoms': ['Sarılık', 'Yorgunluk', 'Bulantı', 'Karın ağrısı', 'Koyu idrar'],
        'causes': ['Hepatit A virüsü', 'Kontamine gıda/su', 'Fekal-oral yol'],
        'treatment': ['Destekleyici bakım', 'İstirahat', 'Sıvı alımı', 'Karaciğer korunması'],
        'prevention': ['Aşılama', 'El hijyeni', 'Güvenli gıda/su', 'Seyahat öncesi aşı'],
        'when_to_see_doctor': ['Sarılık belirtileri', 'Şiddetli karın ağrısı', 'Bilinç değişikliği'],
        'severity': 'Hafif-Orta'
    },
    'Hepatit B': {
        'description': 'Hepatit B virüsünün neden olduğu karaciğer enfeksiyonu.',
        'symptoms': ['Yorgunluk', 'Sarılık', 'Karın ağrısı', 'Koyu idrar', 'Eklem ağrısı'],
        'causes': ['Hepatit B virüsü', 'Kan yoluyla bulaşma', 'Cinsel temas', 'Anne-bebek bulaşması'],
        'treatment': ['Antiviral ilaçlar', 'İmmün modülatörler', 'Karaciğer takibi', 'Yaşam tarzı değişiklikleri'],
        'prevention': ['Aşılama', 'Güvenli cinsel ilişki', 'Steril malzeme kullanımı'],
        'when_to_see_doctor': ['Risk faktörleri varsa', 'Sarılık', 'Sürekli yorgunluk'],
        'severity': 'Orta-Ciddi'
    },
    'Hepatit C': {
        'description': 'Hepatit C virüsünün neden olduğu kronik karaciğer hastalığı.',
        'symptoms': ['Yorgunluk', 'Kas ağrıları', 'İştahsızlık', 'Hafif sarılık', 'Depresyon'],
        'causes': ['Hepatit C virüsü', 'Kan yoluyla bulaşma', 'Steril olmayan enjektör'],
        'treatment': ['Direkt etkili antiviral ilaçlar', 'Karaciğer fonksiyon takibi', 'Yaşam tarzı değişiklikleri'],
        'prevention': ['Güvenli enjektör kullanımı', 'Kan ürünleri taraması', 'Steril malzeme'],
        'when_to_see_doctor': ['Risk faktörleri varsa', 'Test pozitifse', 'Karaciğer hastalığı belirtileri'],
        'severity': 'Ciddi'
    },
    'Hepatit D': {
        'description': 'Hepatit D virüsünün neden olduğu karaciğer enfeksiyonu (Hepatit B ile birlikte).',
        'symptoms': ['Şiddetli yorgunluk', 'Sarılık', 'Karın ağrısı', 'Bulantı', 'Kilo kaybı'],
        'causes': ['Hepatit D virüsü', 'Hepatit B ko-enfeksiyonu', 'Kan yoluyla bulaşma'],
        'treatment': ['Interferon tedavisi', 'Karaciğer desteği', 'Hepatit B tedavisi'],
        'prevention': ['Hepatit B aşısı', 'Güvenli kan ürünleri', 'Steril malzeme kullanımı'],
        'when_to_see_doctor': ['Hepatit B pozitifse', 'Şiddetli karaciğer belirtileri'],
        'severity': 'Ciddi'
    },
    'Hepatit E': {
        'description': 'Hepatit E virüsünün neden olduğu akut karaciğer enfeksiyonu.',
        'symptoms': ['Sarılık', 'Yorgunluk', 'Bulantı', 'Karın ağrısı', 'Ateş'],
        'causes': ['Hepatit E virüsü', 'Kontamine su', 'Kötü sanitasyon'],
        'treatment': ['Destekleyici bakım', 'İstirahat', 'Sıvı elektrolit dengesi'],
        'prevention': ['Temiz su kullanımı', 'Hijyen kuralları', 'Güvenli gıda'],
        'when_to_see_doctor': ['Sarılık belirtileri', 'Hamilelik durumunda', 'Şiddetli semptomlar'],
        'severity': 'Hafif-Orta'
    },
    'Alkolik Hepatit': {
        'description': 'Aşırı alkol tüketiminin neden olduğu karaciğer iltihabı.',
        'symptoms': ['Sarılık', 'Karın şişkinliği', 'Yorgunluk', 'İştahsızlık', 'Ateş'],
        'causes': ['Kronik alkol kullanımı', 'Karaciğer toksisitesi', 'Beslenme yetersizliği'],
        'treatment': ['Alkol bırakma', 'Karaciğer desteği', 'Beslenme desteği', 'Steroid tedavisi'],
        'prevention': ['Alkol tüketimini sınırlama', 'Sağlıklı beslenme', 'Düzenli kontroller'],
        'when_to_see_doctor': ['Alkol sorunu varsa', 'Sarılık', 'Karın şişmesi'],
        'severity': 'Ciddi'
    },
    'Tüberküloz': {
        'description': 'Mycobacterium tuberculosis bakterisinin neden olduğu enfeksiyöz hastalık.',
        'symptoms': ['Sürekli öksürük', 'Balgamda kan', 'Gece terlemesi', 'Kilo kaybı', 'Yorgunluk'],
        'causes': ['Mycobacterium tuberculosis', 'Hava yoluyla bulaşma', 'Düşük bağışıklık'],
        'treatment': ['Çoklu antibiyotik tedavisi', 'DOT (Directly Observed Therapy)', 'İzolasyon'],
        'prevention': ['BCG aşısı', 'Temas takibi', 'Hava kontrolü', 'Bağışıklık güçlendirme'],
        'when_to_see_doctor': ['3 haftadan uzun öksürük', 'Balgamda kan', 'Gece terlemesi'],
        'severity': 'Ciddi'
    },
    'Hemoroid': {
        'description': 'Anüs ve rektum çevresindeki damarların şişmesi ile oluşan hastalık.',
        'symptoms': ['Anüste ağrı', 'Kanama', 'Kaşıntı', 'Şişlik', 'Oturma zorluğu'],
        'causes': ['Kabızlık', 'Hamilelik', 'Uzun süre oturma', 'Ağır kaldırma', 'Yaş'],
        'treatment': ['Lif açısından zengin diyet', 'Merhemler', 'Sıcak oturma banyosu', 'Cerrahi (şiddetli vakalarda)'],
        'prevention': ['Düzenli egzersiz', 'Lif tüketimi', 'Bol su içme', 'Uzun süre oturmaktan kaçınma'],
        'when_to_see_doctor': ['Sürekli kanama', 'Şiddetli ağrı', 'Büyük şişlik', 'Enfeksiyon belirtileri'],
        'severity': 'Hafif-Orta'
    },
    'Varis': {
        'description': 'Bacak damarlarının genişlemesi ve kıvrımlı hale gelmesi.',
        'symptoms': ['Görünür damarlar', 'Bacak ağrısı', 'Şişlik', 'Kramplar', 'Yorgunluk hissi'],
        'causes': ['Genetik faktörler', 'Uzun süre ayakta durma', 'Hamilelik', 'Yaş', 'Obezite'],
        'treatment': ['Kompresyon çorapları', 'Egzersiz', 'Bacakları yüksekte tutma', 'Skleroetrapi', 'Cerrahi'],
        'prevention': ['Düzenli egzersiz', 'Kilo kontrolü', 'Uzun süre aynı pozisyonda kalmaktan kaçınma'],
        'when_to_see_doctor': ['Şiddetli ağrı', 'Ülser oluşumu', 'Kanama', 'Enfeksiyon belirtileri'],
        'severity': 'Hafif-Orta'
    },
    'Hipotiroidizm': {
        'description': 'Tiroid bezinin yetersiz hormon üretimesi ile oluşan hastalık.',
        'symptoms': ['Yorgunluk', 'Kilo alma', 'Soğuk hissetme', 'Kuru cilt', 'Saç dökülmesi'],
        'causes': ['Hashimoto tiroiditi', 'İyot eksikliği', 'Tiroid cerrahisi', 'Radyoaktif iyot tedavisi'],
        'treatment': ['Tiroid hormon replasmanı', 'Levotiroksin', 'Düzenli takip', 'Dozaj ayarlaması'],
        'prevention': ['Yeterli iyot alımı', 'Düzenli kontroller', 'Aile öyküsü varsa tarama'],
        'when_to_see_doctor': ['Sürekli yorgunluk', 'Açıklanamayan kilo alma', 'Soğuk intoleransı'],
        'severity': 'Hafif-Orta'
    },
    'Hipertiroidizm': {
        'description': 'Tiroid bezinin aşırı hormon üretimi ile oluşan hastalık.',
        'symptoms': ['Hızlı kalp atışı', 'Kilo kaybı', 'Aşırı terleme', 'Sinirlilik', 'Titreme'],
        'causes': ['Graves hastalığı', 'Toksik nodüler guatr', 'Tiroidit', 'Aşırı iyot'],
        'treatment': ['Antitiroid ilaçlar', 'Radyoaktif iyot', 'Beta blokerler', 'Cerrahi'],
        'prevention': ['Düzenli kontroller', 'İyot alımını kontrol etme', 'Stres yönetimi'],
        'when_to_see_doctor': ['Hızlı kalp atışı', 'Açıklanamayan kilo kaybı', 'Aşırı terleme'],
        'severity': 'Orta-Ciddi'
    },
    'Hipoglisemi': {
        'description': 'Kan şeker seviyesinin normal değerlerin altına düşmesi.',
        'symptoms': ['Titreme', 'Terleme', 'Çarpıntı', 'Açlık hissi', 'Baş dönmesi'],
        'causes': ['İnsülin aşırı dozu', 'Geç yemek', 'Aşırı egzersiz', 'Alkol', 'Bazı ilaçlar'],
        'treatment': ['Hızlı etkili karbonhidrat', 'Glukagon enjeksiyonu', 'IV glukoz', 'Neden tedavisi'],
        'prevention': ['Düzenli beslenme', 'İlaç dozaj ayarı', 'Kan şekeri takibi', 'Egzersiz planlaması'],
        'when_to_see_doctor': ['Sık hipoglisemi atakları', 'Bilinç kaybı', 'Şiddetli semptomlar'],
        'severity': 'Hafif-Ciddi'
    },
    'Osteoartrit': {
        'description': 'Eklem kıkırdağının aşınması ile oluşan dejeneratif eklem hastalığı.',
        'symptoms': ['Eklem ağrısı', 'Sertlik', 'Şişlik', 'Hareket kısıtlılığı', 'Eklemde ses'],
        'causes': ['Yaşlanma', 'Aşırı kullanım', 'Travma', 'Obezite', 'Genetik faktörler'],
        'treatment': ['Ağrı kesici ilaçlar', 'Fizik tedavi', 'Egzersiz', 'Kilo verme', 'Eklem enjeksiyonları'],
        'prevention': ['Düzenli egzersiz', 'Kilo kontrolü', 'Eklem korunması', 'Doğru postür'],
        'when_to_see_doctor': ['Şiddetli ağrı', 'Hareket kaybı', 'Günlük yaşamı etkileme', 'Şişlik'],
        'severity': 'Hafif-Orta'
    },
    'Artrit': {
        'description': 'Eklemlerin iltihaplanması ile karakterize hastalık grubu.',
        'symptoms': ['Eklem ağrısı', 'Şişlik', 'Kızarıklık', 'Ateş', 'Hareket kısıtlılığı'],
        'causes': ['Otoimmün nedenler', 'Enfeksiyonlar', 'Kristal birikimi', 'Travma'],
        'treatment': ['Antiinflamatuar ilaçlar', 'Hastalık modifiye edici ilaçlar', 'Fizik tedavi', 'Egzersiz'],
        'prevention': ['Düzenli egzersiz', 'Sağlıklı beslenme', 'Kilo kontrolü', 'Stres yönetimi'],
        'when_to_see_doctor': ['Şiddetli eklem ağrısı', 'Şişlik', 'Hareket kaybı', 'Sistemik belirtiler'],
        'severity': 'Hafif-Ciddi'
    },
    'Vertigo': {
        'description': 'Baş dönmesi ve denge kaybı ile karakterize vestibuler bozukluk.',
        'symptoms': ['Döner tarzda baş dönmesi', 'Bulantı', 'Kusma', 'Denge kaybı', 'Kulak çınlaması'],
        'causes': ['İç kulak problemleri', 'Benign pozisyonel vertigo', 'Meniere hastalığı', 'Vestibuler nörit'],
        'treatment': ['Pozisyonel manevralar', 'Antivertijinöz ilaçlar', 'Vestibuler rehabilitasyon', 'Altta yatan neden tedavisi'],
        'prevention': ['Ani baş hareketlerinden kaçınma', 'Stres yönetimi', 'Düzenli egzersiz'],
        'when_to_see_doctor': ['Şiddetli vertigo', 'İşitme kaybı', 'Başağrısı', 'Nörolojik belirtiler'],
        'severity': 'Hafif-Orta'
    },
    'Akne': {
        'description': 'Yağ bezlerinin tıkanması sonucu oluşan deri hastalığı.',
        'symptoms': ['Sivilceler', 'Siyah noktalar', 'Beyaz noktalar', 'Iltihaplı nodüller', 'Skar oluşumu'],
        'causes': ['Hormon değişiklikleri', 'Aşırı yağ üretimi', 'Bakteriyel enfeksiyon', 'Genetik faktörler'],
        'treatment': ['Topikal retinoidler', 'Antibiyotik kremler', 'Benzol peroksit', 'Oral antibiyotikler'],
        'prevention': ['Düzenli temizlik', 'Uygun kozmetik ürünler', 'Sivilce sıkmaktan kaçınma'],
        'when_to_see_doctor': ['Şiddetli akne', 'Skar oluşumu', 'Psikolojik etkilenme', 'Direnç durumu'],
        'severity': 'Hafif-Orta'
    },
    'İdrar Yolu Enfeksiyonu': {
        'description': 'İdrar yollarında bakteriyel enfeksiyon oluşumu.',
        'symptoms': ['İdrar yaparken yanma', 'Sık idrara çıkma', 'Acil idrar hissi', 'Bulanık idrar', 'Pelvik ağrı'],
        'causes': ['E. coli bakterisi', 'Cinsel aktivite', 'Kadın anatomisi', 'İdrar tutma', 'Bağışıklık azalması'],
        'treatment': ['Antibiyotik tedavisi', 'Bol sıvı tüketimi', 'Ağrı kesiciler', 'Cranberry suyu'],
        'prevention': ['Bol su içme', 'Düzenli boşaltım', 'İdrar sonrası temizlik', 'Pamuklu iç giyim'],
        'when_to_see_doctor': ['Ateş', 'Sırt ağrısı', 'Şiddetli semptomlar', 'Tekrarlayan enfeksiyonlar'],
        'severity': 'Hafif-Orta'
    },
    'Sedef Hastalığı': {
        'description': 'Kronik, otoimmün karakterli deri hastalığı.',
        'symptoms': ['Gümüşi pullu plaklar', 'Kızarıklık', 'Kaşıntı', 'Cilt kalınlaşması', 'Eklem ağrısı'],
        'causes': ['Otoimmün faktörler', 'Genetik yatkınlık', 'Stres', 'Enfeksiyonlar', 'Çevresel faktörler'],
        'treatment': ['Topikal kortikosteroidler', 'Vitamin D analogları', 'Fototerapi', 'Sistemik ilaçlar'],
        'prevention': ['Stres yönetimi', 'Cilt nemlendirilmesi', 'Tetikleyicilerden kaçınma', 'Düzenli takip'],
        'when_to_see_doctor': ['Yaygın tutulum', 'Eklem ağrısı', 'Psikolojik etkilenme', 'Tedaviye direnç'],
        'severity': 'Hafif-Orta'
    },
    'İmpetigo': {
        'description': 'Bakteriyel karakterde yüzeyel deri enfeksiyonu.',
        'symptoms': ['Su dolu kabarcıklar', 'Bal rengi kabuklar', 'Kaşıntı', 'Kızarıklık', 'Yayılma eğilimi'],
        'causes': ['Staphylococcus aureus', 'Streptococcus pyogenes', 'Cilt travması', 'Kötü hijyen'],
        'treatment': ['Topikal antibiyotikler', 'Oral antibiyotikler', 'Antiseptik solüsyonlar', 'Hijyen önlemleri'],
        'prevention': ['El hijyeni', 'Yaraların temizlenmesi', 'Kişisel eşya paylaşmama', 'Temiz giysi'],
        'when_to_see_doctor': ['Yaygın lezyonlar', 'Ateş', 'Lenfadenopati', 'Tedaviye yanıtsızlık'],
        'severity': 'Hafif'
    }
}

def get_disease_info(disease_name):
    """Hastalık bilgilerini getir"""
    return disease_info.get(disease_name, {
        'description': 'Bu hastalık hakkında detaylı bilgi henüz mevcut değil.',
        'symptoms': ['Belirti bilgisi mevcut değil'],
        'causes': ['Neden bilgisi mevcut değil'],
        'treatment': ['Tedavi bilgisi mevcut değil'],
        'prevention': ['Önlem bilgisi mevcut değil'],
        'when_to_see_doctor': ['Doktora başvuru zamanı bilgisi mevcut değil'],
        'severity': 'Bilinmiyor'
    })

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Detaylı PDF rapor oluştur"""
    try:
        data = request.json
        prediction_data = data.get('prediction_data', {})
        selected_symptoms = data.get('symptoms', [])

        if not prediction_data:
            return jsonify({'error': 'Rapor oluşturmak için önce teşhis yapmalısınız'}), 400

        # Rapor içeriği oluştur
        report_content = {
            'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M'),
            'patient_symptoms': [get_turkish_symptom_name(symptom) for symptom in selected_symptoms],
            'diagnosis': prediction_data.get('prediction', ''),
            'model_used': prediction_data.get('model_used', ''),
            'probabilities': prediction_data.get('probabilities', []),
            'disease_info': prediction_data.get('disease_info', {}),
            'report_id': f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        return jsonify({
            'success': True,
            'report': report_content,
            'message': 'Rapor başarıyla oluşturuldu'
        })

    except Exception as e:
        return jsonify({'error': f'Rapor oluşturma hatası: {str(e)}'}), 500

@app.route('/chat-process', methods=['POST'])
def chat_process():
    """Chatbot mesajlarını işleyen endpoint"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        conversation_history = data.get('conversation_history', [])

        if not message:
            return jsonify({'error': 'Mesaj boş olamaz'}), 400

        # Basit semptom çıkarma (keyword-based)
        extracted_symptoms = extract_symptoms_from_text(message)

        # Yanıt oluştur
        if extracted_symptoms:
            response = {
                'type': 'symptoms_detected',
                'message': f"Şu semptomları tespit ettim: {', '.join([get_turkish_symptom_name(s) for s in extracted_symptoms])}",
                'symptoms': extracted_symptoms,
                'suggestions': get_follow_up_questions(extracted_symptoms)
            }
        elif any(keyword in message.lower() for keyword in ['teşhis', 'analiz', 'tani', 'sonuç']):
            response = {
                'type': 'diagnosis_request',
                'message': 'Semptomlarınızı belirttiniz. Şimdi analiz yapabilirsiniz.',
                'action': 'proceed_to_diagnosis'
            }
        else:
            response = {
                'type': 'clarification',
                'message': 'Anlayamadım, semptomlarınızı daha detaylı anlatabilir misiniz? ',
                'suggestions': [
                    "Baş ağrım var",
                    "Mide bulantım var",
                    "Öksürüyorum",
                    "Cildimde döküntü var",
                ]
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Chat işleme hatası: {str(e)}'}), 500

def extract_symptoms_from_text(text):
    """Metinden semptom çıkarma fonksiyonu"""
    detected_symptoms = []
    text_lower = text.lower()

    # SEMPTOM-KEYWORD SÖZLÜĞÜ
    symptom_keywords = {
        'itching': ['kaşın', 'kaşı', 'kaşıntı', 'kaşık'],
        'skin_rash': ['cilt döküntü', 'döküntü', 'kızarık', 'cilt kızar', 'döküntülü'],
        'nodal_skin_eruptions': ['nodül', 'cilt nodül', 'şişlik cilt', 'cilt çıkıntı'],
        'continuous_sneezing': ['sürekli hapşı', 'hapşır', 'hapşırma', 'aksır'],
        'shivering': ['titre', 'titrek', 'sallan'],
        'chills': ['üşü', 'üşüme', 'soğuk'],
        'joint_pain': ['eklem ağrı', 'kemik ağrı', 'eklem', 'romatizma', 'kemikler ağrı'],
        'stomach_pain': ['mide ağrı', 'karın ağrı', 'mide yanı', 'mide acı'],
        'acidity': ['ekşi', 'mide ekşi', 'yanma mide', 'asit'],
        'ulcers_on_tongue': ['dil yara', 'dil ülser', 'dil aft', 'dilde yara'],
        'muscle_wasting': ['kas eri', 'kas zayıf', 'kas kayb'],
        'vomiting': ['kus', 'kusmak', 'kusma', 'mide boşalt'],
        'burning_micturition': ['idrar yan', 'işer yanma', 'idrar yakar'],
        'spotting_ urination': ['kanla idrar', 'idrar kan', 'kanlı idrar'],
        'fatigue': ['yorgun', 'bitkin', 'halsiz', 'güçsüz', 'enerjisiz'],
        'weight_gain': ['kilo al', 'şişmanla', 'kilo art'],
        'anxiety': ['anksiye', 'kaygı', 'endişe', 'stres', 'gergin'],
        'cold_hands_and_feets': ['soğuk el ayak', 'el ayak soğuk', 'soğuk uzuv'],
        'mood_swings': ['ruh hali değiş', 'duygusal', 'sinirli'],
        'weight_loss': ['kilo kay', 'zayıfla', 'kilo düş'],
        'restlessness': ['huzursuz', 'rahat', 'sakin'],
        'lethargy': ['uyuşuk', 'durgun', 'tembellik'],
        'patches_in_throat': ['boğaz leke', 'boğazda leke', 'boğaz beyaz'],
        'irregular_sugar_level': ['şeker düzensiz', 'kan şeker', 'şeker seviye'],
        'cough': ['öksürük', 'öksür', 'öksüren', 'kuru öksürük'],
        'high_fever': ['yüksek ateş', 'şiddetli ateş', 'ateş yüksek'],
        'sunken_eyes': ['çökük göz', 'göz çukur', 'gözler çukur'],
        'breathlessness': ['nefes dar', 'nefes alam', 'solun', 'nefes kesi'],
        'sweating': ['terle', 'aşırı ter', 'ter dökmek'],
        'dehydration': ['susuz', 'kuru', 'sıvı kay'],
        'indigestion': ['hazımsız', 'sindir', 'mide bozu'],
        'headache': ['baş ağrı', 'başım ağrı', 'migren', 'baş ağır'],
        'yellowish_skin': ['sarı cilt', 'sarımsı', 'sarılık'],
        'dark_urine': ['koyu idrar', 'idrar koyu', 'idrar renk'],
        'nausea': ['bulantı', 'mide bulantı', 'bulant'],
        'loss_of_appetite': ['iştah', 'yemek iste', 'açlık yok'],
        'pain_behind_the_eyes': ['göz arkası ağrı', 'göz arka', 'göz çevre ağrı'],
        'back_pain': ['sırt ağrı', 'bel ağrı', 'sırtım ağrı', 'omurga ağrı'],
        'constipation': ['kabız', 'tuvalete çıkam', 'dışkı sert'],
        'abdominal_pain': ['karın ağrı', 'karnım ağrı', 'karın acı'],
        'diarrhoea': ['ishal', 'karın boşal', 'sulu dışkı', 'amel'],
        'mild_fever': ['hafif ateş', 'düşük ateş', 'ateş az'],
        'yellow_urine': ['sarı idrar', 'idrar sarı'],
        'yellowing_of_eyes': ['göz sarı', 'gözde sarılık', 'göz sarıla'],
        'acute_liver_failure': ['karaciğer yetmez', 'karaciğer kriz'],
        'fluid_overload': ['sıvı fazla', 'su toplan', 'ödem'],
        'swelling_of_stomach': ['mide şiş', 'karın şiş', 'mide bombe'],
        'swelled_lymph_nodes': ['lenf şiş', 'bezler şiş', 'boyun şiş'],
        'malaise': ['halsiz', 'hasta hiss', 'rahatsız'],
        'blurred_and_distorted_vision': ['bulanık görme', 'görme bozuk', 'göz bulanık'],
        'phlegm': ['balgam', 'tükür', 'göğüs balgam'],
        'throat_irritation': ['boğaz tahriş', 'boğaz yanma', 'boğaz kaşın'],
        'redness_of_eyes': ['göz kızar', 'gözde kızarık', 'göz al'],
        'sinus_pressure': ['sinüs basın', 'burun basın', 'alın basın'],
        'runny_nose': ['burun ak', 'burun sızın', 'nezle'],
        'congestion': ['tıkan', 'burun tıkan', 'sinüs tıkan'],
        'chest_pain': ['göğüs ağrı', 'göğsüm ağrı', 'kalp ağrı', 'göğüs sıkış'],
        'weakness_in_limbs': ['uzuv güçsüz', 'kol bacak güçsüz', 'uzuvlarda güçsüz'],
        'fast_heart_rate': ['kalp hızlı', 'nabız hızlı', 'çarpıntı'],
        'pain_during_bowel_movements': ['tuvalet ağrı', 'dışkı ağrı', 'büyük abdest ağrı'],
        'pain_in_anal_region': ['makattan ağrı', 'kuyruk ağrı', 'anüs ağrı'],
        'bloody_stool': ['kanlı dışkı', 'dışkı kan', 'kakada kan'],
        'irritation_in_anus': ['makattan kaşın', 'pişik', 'anüs tahriş'],
        'neck_pain': ['boyun ağrı', 'boynum ağrı', 'ense ağrı'],
        'dizziness': ['baş dön', 'sersem', 'dengem', 'vertigo'],
        'cramps': ['kramp', 'kas spazmı', 'kasılma'],
        'bruising': ['mor', 'çürük', 'berelenme'],
        'obesity': ['şişman', 'obez', 'kilolu'],
        'swollen_legs': ['bacak şiş', 'ayak şiş', 'bacaklarda şiş'],
        'swollen_blood_vessels': ['damar şiş', 'toplardamar şiş', 'varis'],
        'puffy_face_and_eyes': ['yüz şiş', 'göz şiş', 'yüzde şiş'],
        'enlarged_thyroid': ['tiroid büyük', 'guatr', 'boyun şiş'],
        'brittle_nails': ['tırnak kırıl', 'tırnak zayıf', 'tırnaklar kırık'],
        'swollen_extremeties': ['uzuv şiş', 'el ayak şiş', 'ekstremite şiş'],
        'excessive_hunger': ['aşırı açlık', 'çok aç', 'sürekli aç'],
        'extra_marital_contacts': ['evlilik dışı', 'güvenli olmayan ilişki'],
        'drying_and_tingling_lips': ['dudak kuru', 'dudak karınca', 'dudaklarda uyuşma'],
        'slurred_speech': ['konuşma bozuk', 'peltek', 'dilim dolaş'],
        'knee_pain': ['diz ağrı', 'dizim ağrı', 'diz acı'],
        'hip_joint_pain': ['kalça ağrı', 'kalçam ağrı', 'kalça eklem'],
        'muscle_weakness': ['kas güçsüz', 'kas zayıf', 'kaslar güçsüz'],
        'stiff_neck': ['boyun sert', 'boyun katı', 'ense sert'],
        'swelling_joints': ['eklem şiş', 'eklemlerde şiş', 'kemik şiş'],
        'movement_stiffness': ['hareket güç', 'katılık', 'sertlik'],
        'spinning_movements': ['sersemlik', 'baş dönme', 'döner'],
        'loss_of_balance': ['denge kay', 'dengesiz', 'düş'],
        'unsteadiness': ['dengesiz', 'sallan', 'titrek'],
        'weakness_of_one_body_side': ['yarım vücut güçsüz', 'tek taraf zayıf', 'felç'],
        'loss_of_smell': ['koku alamam', 'koku kay', 'koklama yok'],
        'bladder_discomfort': ['mesane rahatsız', 'idrar kesesi', 'mesane ağrı'],
        'foul_smell_of urine': ['idrar koku', 'idrar pis koku', 'kötü kokulu idrar'],
        'continuous_feel_of_urine': ['sürekli idrar hiss', 'idrar hiss', 'tuvalete gidesi'],
        'passage_of_gases': ['gaz', 'osuruk', 'bağırsak gaz'],
        'internal_itching': ['iç kaşın', 'içten kaşın', 'iç taraf kaşın'],
        'toxic_look_(typhos)': ['hasta görün', 'zehirli görün', 'solgun'],
        'depression': ['depres', 'üzün', 'mutsuz', 'çökük'],
        'irritability': ['sinirli', 'çabuk sinirlen', 'asabi'],
        'muscle_pain': ['kas ağrı', 'kaslarım ağrı', 'kas acı'],
        'altered_sensorium': ['kafa karışık', 'bilinç bulanık', 'farkında değil'],
        'red_spots_over_body': ['vücutta kırmızı leke', 'kırmızı nokta', 'döküntü kırmızı'],
        'belly_pain': ['karın ağrı', 'göbek ağrı', 'belly ağrı'],
        'abnormal_menstruation': ['adet düzensiz', 'regl problemli', 'adet bozuk'],
        'dischromic _patches': ['cilt renk değiş', 'leke cilt', 'renk farkı'],
        'watering_from_eyes': ['göz sular', 'gözyaşı', 'gözden su gelme'],
        'increased_appetite': ['iştah art', 'çok yemek iste', 'aşırı iştah'],
        'polyuria': ['sık idrar', 'çok idrar çıkar', 'aşırı idrar'],
        'family_history': ['aile geçmiş', 'ailede hasta', 'kalıtsal'],
        'mucoid_sputum': ['mukozlu balgam', 'sümüklü balgam', 'yapışkan balgam'],
        'rusty_sputum': ['paslı balgam', 'kahverengi balgam', 'kanlı balgam'],
        'lack_of_concentration': ['konsantrasyon yok', 'dikkatsiz', 'odaklan'],
        'visual_disturbances': ['görme bozuk', 'görme problemi', 'göz problemi'],
        'receiving_blood_transfusion': ['kan nakli', 'kan almak', 'transfüzyon'],
        'receiving_unsterile_injections': ['steril olmayan iğne', 'kirli enjektör'],
        'coma': ['koma', 'bayılma', 'bilinç kay'],
        'stomach_bleeding': ['mide kana', 'mide kan', 'gastrik kanama'],
        'distention_of_abdomen': ['karın şişkin', 'karında şişme', 'batın şiş'],
        'history_of_alcohol_consumption': ['alkol geçmiş', 'içki kullan', 'alkol'],
        'fluid_overload.1': ['sıvı yüklen', 'su fazla', 'sıvı birikim'],
        'blood_in_sputum': ['balgam kan', 'kan tükür', 'balgamda kan'],
        'prominent_veins_on_calf': ['baldır damar', 'bacak damar belirgin', 'varis'],
        'palpitations': ['çarpıntı', 'kalp hızlı', 'kalp çarp'],
        'painful_walking': ['yürürken ağrı', 'yürüme ağrı', 'adım ağrı'],
        'pus_filled_pimples': ['irinli sivilce', 'irin', 'sivilce irin'],
        'blackheads': ['siyah nokta', 'komedon', 'siyah ben'],
        'scurring': ['kabuk', 'yara kabuğu', 'kuruma'],
        'skin_peeling': ['cilt soyul', 'deri soyul', 'kabuklan'],
        'silver_like_dusting': ['gümüş toz', 'pul pul', 'kepek'],
        'small_dents_in_nails': ['tırnak çukur', 'tırnakta nokta', 'tırnak delik'],
        'inflammatory_nails': ['tırnak iltihap', 'tırnak enfeksiyon', 'tırnak kızar'],
        'blister': ['su toplan', 'kabarcık', 'blister'],
        'red_sore_around_nose': ['burun çevre yara', 'burun kızar', 'burun yara'],
        'yellow_crust_ooze': ['sarı kabuk', 'sarı akıntı', 'irin akma'],
    }

    # Keyword eşleştirme
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                if symptom not in detected_symptoms:
                    detected_symptoms.append(symptom)
                break

    return detected_symptoms

def get_follow_up_questions(symptoms):
    """Semptomları için takip soruları önerisi"""
    questions = []

    if 'fever' in symptoms:
        questions.append("Ateşiniz ne kadar süredir devam ediyor?")
        questions.append("Ateş dereceniz yaklaşık kaç?")

    if 'headache' in symptoms:
        questions.append("Baş ağrınız hangi bölgede?")
        questions.append("Ağrı zonklayıcı mı yoksa baskı hissi mi?")

    if 'stomach_pain' in symptoms:
        questions.append("Ağrı yemeklerden önce mi sonra mı artıyor?")
        questions.append("Karın ağrınız nerede tam olarak?")

    if not questions:
        questions = [
            "Başka semptomlarınız var mı?",
        ]

    return questions[:2]  # En fazla 2 soru

if __name__ == '__main__':
    # Modelleri yükle
    load_models_and_data()

    # Uygulamayı çalıştır
    app.run(debug=True, host='0.0.0.0', port=5000)
