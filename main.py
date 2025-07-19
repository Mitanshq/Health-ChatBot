# C:\Users\DELL\AppData\Local\Programs\Python\Python313\python.exe -m pip install

import spacy
import pandas as pd
import contractions
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os, json
import threading

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

text_to_intent_file = 'databases/intent.csv'
intent_to_disease_file = 'databases/symptoms_to_disease.csv'
disease_to_cure_file = 'databases/disease_to_cure.csv'

SYMPTOM_SYNONYMS = {
    # Fever-related
    "feverish": "fever",
    "high temperature": "fever",
    "chills": "fever",
    "hot body": "fever",
    
    # Headache-related
    "head pain": "headache",
    "head is hurting": "headache",
    "head is paining": "headache",
    "throbbing head": "headache",

    # Fatigue-related
    "tired": "fatigue",
    "exhausted": "fatigue",
    "weak": "fatigue",
    "loss of energy": "fatigue",
    "sluggish": "fatigue",

    # Cough-related
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "continuous cough": "cough",
    "throat irritation": "cough",

    # Cold-related
    "runny nose": "cold",
    "blocked nose": "cold",
    "nasal congestion": "cold",
    "sneezing": "cold",

    # Sore throat-related
    "throat pain": "sore throat",
    "painful swallowing": "sore throat",
    "scratchy throat": "sore throat",

    # Shortness of breath-related
    "difficulty breathing": "shortness of breath",
    "breathless": "shortness of breath",
    "can't breathe": "shortness of breath",

    # Body ache-related
    "body hurts": "body pain",
    "muscle ache": "body pain",
    "joint pain": "body pain",
    "aching body": "body pain",

    # Stomach-related
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "cramps": "abdominal pain",
    "tummy pain": "abdominal pain",

    # Diarrhea-related
    "loose motion": "diarrhea",
    "watery stool": "diarrhea",
    "frequent bowel movement": "diarrhea",

    # Vomiting-related
    "puking": "vomiting",
    "throwing up": "vomiting",
    "nauseated": "nausea",

    # Dizziness-related
    "lightheaded": "dizziness",
    "spinning head": "dizziness",
    "feeling faint": "dizziness",

    # Chest pain-related
    "chest tightness": "chest pain",
    "chest discomfort": "chest pain",

    # Rash-related
    "red spots": "rash",
    "itchy skin": "rash",
    "skin bumps": "rash"
}

SEVERITY_SYNONYMS = {
    "very bad": "severe",
    "extreme": "severe",
    "a lot": "severe",
    "too much": "severe",
    "intense": "severe",
    "mild": "mild",
    "slight": "mild",
    "bit of": "mild",
    "small": "mild",
    "very much": "severe",
    "very less": "mild"
}


def normalize_query(query):
    # 1. Lowercase and expand contractions
    query = contractions.fix(query)

    # 2. Replace known symptom phrases
    for phrase, replacement in SYMPTOM_SYNONYMS.items():
        query = query.replace(phrase, replacement)
        
    for word, rep in SEVERITY_SYNONYMS.items():
        query = query.replace(word, rep)

    # 3. Process with spaCy
    doc = nlp(query)

    # 4. Lemmatize and remove stopwords and punctuation
    clean_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    return " ".join(clean_tokens)

def train_model():
    text2intentData = pd.read_csv(text_to_intent_file)
    intent2diseaseData = pd.read_csv(intent_to_disease_file)
    

    text2intentData.dropna()
    intent2diseaseData.dropna()

    intent2diseaseData['symptoms'] = intent2diseaseData['symptoms'].str.lower().str.replace(',', ' ')


    le = LabelEncoder()
    intent2diseaseData['label'] = le.fit_transform(intent2diseaseData['disease'])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(intent2diseaseData['symptoms'])
    y = intent2diseaseData['label']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    
    return model ,vectorizer, le

files = ['models/disease_model.pkl', 'models/vectorizer.pkl', 'models/label_encoder.pkl']
os.makedirs('models', exist_ok=True)

if all(os.path.exists(f) for f in files):
    # ✅ Load all
    with open('models/disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
else:
    # ❌ Train and Save (you already trained above)
    print("no file saved")
    model, vectorizer, le = train_model()
    train_model()
    with open('models/disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
Q_TABLE_FILE = 'reinforcement_memory.json'
def reinforcement_learning(symptoms, cure, feedback):
    state = ",".join(symptoms).lower()
    action = cure.lower()
    reward = 1 if feedback in ["yes", "y", "correct"] else -1
    
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'r') as f:
            q_table = json.load(f)
    else:
        q_table = {}
        
    key = (state, action)
    key_str = '|||'.join(key)
    
    if key_str in q_table:
        q_table[key_str] += reward
    else:
        q_table[key_str] = reward
        
    # save q_file
    with open(Q_TABLE_FILE, 'w') as f:
        json.dump(q_table, f, indent=2)
    
def apply_renf_learning():
    # this will apply reinforcement learning to the actual model and training dataset
    if not os.path.exists(Q_TABLE_FILE):
        print("⚠️ Reinforcement file not found.")
        return

    with open(Q_TABLE_FILE, 'r') as f:
        data = json.load(f)
        
    # filtering good data
    positive_data = [
        (k.split("|||")[0], k.split("|||")[1])
        for k, v in data.items()
        if v > 0.5
    ]
    if not positive_data:
        return
    
    # convert to dataframe
    feedback_df = pd.DataFrame(positive_data, columns=['symptoms', 'disease'])
    
    # load original training data
    df_ini = pd.read_csv(intent_to_disease_file)
    
    # combine both df
    df_combined = pd.concat([df_ini, feedback_df], ignore_index=True).drop_duplicates()
    
    # prepare training features
    df_combined['symptoms'] = df_combined['symptoms'].str.lower().str.replace(",", " ")
    df_combined['disease'] = df_combined['disease'].str.lower()
    X_raw = df_combined['symptoms']
    y_raw = df_combined['disease']
    
    # re vectoerize and re train
    vectorizer_new = TfidfVectorizer()
    X = vectorizer_new.fit_transform(X_raw)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    model_new = RandomForestClassifier()
    model_new.fit(X, y)

    # Save new model and encoders
    with open('models/disease_model.pkl', 'wb') as f:
        pickle.dump(model_new, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer_new, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Optional: clear feedback after learning
    with open(Q_TABLE_FILE, 'w') as f:
        json.dump({}, f)

def predict_disease(query, threshold=0.2,top_n=3 ):
    norm = normalize_query(", ".join(query) if isinstance(query, list) else query)
    vec = vectorizer.transform([norm])
    probs  = model.predict_proba(vec)[0]
    
    top_indices = probs.argsort()[-top_n:][::-1]
    top_diseases = [(le.inverse_transform([i])[0], probs[i]) for i in top_indices]
    
    
    if top_diseases[0][1] - top_diseases[1][1] < threshold:
        return {
            "status": "ambiguous",
            "possible_diseases": top_diseases,
            "message": "Your symptoms match multiple conditions. Can you be more specific?"
        }
    else:
        return {
            "status": "confident",
            "predicted": top_diseases[0][0],
            "confidence": top_diseases[0][1]
        }
        
def predict_cure(disease):
    disease = disease.lower()
    disease2cure = pd.read_csv(disease_to_cure_file)
    disease2cure['disease'] = disease2cure['disease'].str.lower()

    filtered = disease2cure[disease2cure['disease'] == disease]

    if filtered.empty:
        return {
            "medication": ["No medication info found"],
            "remedies": ["No remedies info found"]
        }

    # Drop duplicates and nulls
    medications = filtered['medication'].dropna().drop_duplicates().tolist()
    remedies = filtered['remedy'].dropna().drop_duplicates().tolist()

    return {
        "medication": medications,
        "remedies": remedies
    }

    
def process_symp(symptoms):
    prediction = predict_disease(symptoms)
    
    if prediction['status'] == 'ambiguous':
        return prediction

    cure = predict_cure(prediction['predicted'])
    
    return {
        "status": "confident",
        "predicted": prediction['predicted'],
        "confidence": round(prediction['confidence'], 2),
        "medication": cure["medication"],
        "remedies": cure["remedies"]
    }
    
