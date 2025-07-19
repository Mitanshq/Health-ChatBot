from flask import Flask, render_template, request, jsonify, session
from main import predict_disease, predict_cure, reinforcement_learning, apply_renf_learning
from difflib import get_close_matches
import threading, os, json, time, atexit, datetime

app = Flask(__name__)
app.secret_key = "secret"

CHAT_FILE = 'chats.json'

negative_phrases = [
    'no', 'nah', 'nope', 'not really', 'i am fine', 'nothing', 'np',
    'thank you', 'you are right', 'correct', 'right', 'no problem'
]

known_diseases = {
    "malaria": ["high fever", "chills", "sweating", "headache", "nausea", "vomiting"],
    "dengue": ["high fever", "rash", "headache", "muscle pain", "joint pain", "pain behind eyes"],
    "diabetes": ["frequent urination", "excessive thirst", "unexplained weight loss", "fatigue"],
    "asthma": ["wheezing", "shortness of breath", "chest tightness", "coughing at night"],
    "covid-19": ["fever", "dry cough", "tiredness", "loss of taste", "difficulty breathing"],
    "cold": ["runny nose", "sore throat", "sneezing", "mild headache", "low-grade fever"],
    "flu": ["high fever", "cough", "muscle pain", "fatigue", "chills"],
    "typhoid": ["high fever", "abdominal pain", "weakness", "constipation", "loss of appetite"],
    "jaundice": ["yellowing skin", "dark urine", "fatigue", "abdominal pain", "nausea"],
    "tuberculosis": ["chronic cough", "weight loss", "night sweats", "fever", "blood in sputum"],
    "pneumonia": ["chest pain", "shortness of breath", "cough with phlegm", "fever"],
    "bronchitis": ["cough with mucus", "wheezing", "chest discomfort", "fatigue"],
    "migraine": ["severe headache", "nausea", "sensitivity to light", "blurred vision"],
    "anemia": ["fatigue", "pale skin", "shortness of breath", "dizziness", "cold hands and feet"],
    "chickenpox": ["itchy rash", "fever", "tiredness", "loss of appetite"],
    "measles": ["rash", "fever", "runny nose", "red eyes", "cough"],
    "hypertension": ["headache", "dizziness", "blurred vision", "nosebleeds", "fatigue"],
    "hypothyroidism": ["fatigue", "weight gain", "cold sensitivity", "depression", "dry skin"],
    "appendicitis": ["abdominal pain", "nausea", "vomiting", "loss of appetite", "fever"],
    "urinary tract infection": ["burning urination", "frequent urination", "cloudy urine", "pelvic pain"]
}


def disease_to_sypmtom(disease):
    return known_diseases.get(disease.lower())

def find_closest_match(user_ip):
    matches = get_close_matches(user_ip, known_diseases.keys(), n=1, cutoff=0.8)
    return matches[0] if matches else None

@app.route('/')
def home():
    session['symptoms'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "").strip().lower()
    
    disease = find_closest_match(user_input)

    if user_input in known_diseases:
        symptoms = disease_to_sypmtom(user_input)
        cure = predict_cure(disease)
        messages = [
            f"You mentioned {disease}",
            f"Here are some Symptoms: {', '.join(set(symptoms))}",
            f"Suggested Medications: {', '.join(set(cure['medication']))}",
            f"Home remedies: {', '.join(set(cure['remedies']))}",
            "Let me know if you want to ask about another disease or have symptoms to share."
        ]
        for msg in messages:
            session["chat_session"].append({"role": "bot", "text": msg})
        return jsonify({"messages": messages})
    
    # Handle feedback
    if user_input in ["yes", "y", "correct", "no", "n", "wrong", "incorrect"]:
        predicted_data = session.get('last_prediction')
        if predicted_data:
            reinforcement_learning(predicted_data['symptoms'], predicted_data['disease'], user_input)
            threading.Thread(target=apply_renf_learning, daemon=True).start()
            session.pop("last_prediction", None)
            messages = [
                "✅ Feedback saved.",
                "You can now start a new session by entering your symptoms or asking about any disease."
            ]
            for msg in messages:
                session["chat_session"].append({"role": "bot", "text": msg})
            return jsonify({"messages": messages})

    # Reset on negatives
    if user_input in negative_phrases:
        session['symptoms'] = []
        messages = [
            "🟢 Session reset.",
            "Tell me your new symptoms so I can help you better!"
        ]
        for msg in messages:
            session["chat_session"].append({"role": "bot", "text": msg})
        return jsonify({"messages": messages})

    # Small talk
    greetings = ["hi", "hello", "hey", "good morning", "good evening", 'gm', 'wassup', 'hey there']
    gratitude = ["thanks", "thank you", "thnx", "thank u", "ty", 'tysm']
    polite_words = ["please", "okay", "ok", "hmm", "cool", 'pls', 'plss']

    if user_input in greetings:
        msg = "👋 Hello! I'm your health assistant bot. Tell me your symptoms, and I’ll try to help."
        session["chat_session"].append({"role": "bot", "text": msg})
        return jsonify({"messages": [msg]})

    if user_input in gratitude:
        msg = "😊 You're welcome! Let me know if you have any symptoms or questions."
        session["chat_session"].append({"role": "bot", "text": msg})
        return jsonify({"messages": [msg]})

    if user_input in polite_words:
        msg = "👍 Got it! Please go ahead and share your symptom."
        session["chat_session"].append({"role": "bot", "text": msg})
        return jsonify({"messages": [msg]})

    # Symptom accumulation
    symptoms = session.get('symptoms', [])
    symptoms.append(user_input)
    session['symptoms'] = symptoms

    response = predict_disease(", ".join(symptoms))

    if response['status'] == 'ambiguous':
        top_diseases = [f"{d}" for d, _ in response['possible_diseases']]
        messages = [
            "🤔 I'm not fully confident yet.",
            f"Possible conditions include: {', '.join(top_diseases)}.",
            "Can you tell me a few more symptoms?"
        ]
        return jsonify({"messages": messages})

    elif response['status'] == 'confident':
        disease = response['predicted']
        cure = predict_cure(disease)
        messages = [
            f"✅ I believe you might have {disease}",
            "💊 Recommended Medications: " + ", ".join(set(cure['medication'])),
            "🏡 Home Remedies: " + ", ".join(set(cure['remedies'])),
            "Was this helpful? (yes/no)"
        ]

        session['last_prediction'] = {
            "symptoms": symptoms.copy(),
            "disease": disease
        }
        session['symptoms'] = []
        return jsonify({"messages": messages})

    # Error fallback
    msg = "❌ Sorry, something went wrong. Please try again."
    return jsonify({"messages": [msg]})




if __name__ == '__main__':
    app.run(debug=True)
