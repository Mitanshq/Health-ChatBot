from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from main import predict_disease, predict_cure, reinforcement_learning, apply_renf_learning
from difflib import get_close_matches
import threading, os, bcrypt
from sqlite3 import *

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
    return render_template('index.html', user=session.get("user"))

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
            return jsonify({"messages": messages})

    # Reset on negatives
    if user_input in negative_phrases:
        session['symptoms'] = []
        messages = [
            "🟢 Session reset.",
            "Tell me your new symptoms so I can help you better!"
        ]
        return jsonify({"messages": messages})

    # Small talk
    greetings = ["hi", "hello", "hey", "good morning", "good evening", 'gm', 'wassup', 'hey there']
    gratitude = ["thanks", "thank you", "thnx", "thank u", "ty", 'tysm']
    polite_words = ["please", "okay", "ok", "hmm", "cool", 'pls', 'plss']

    if user_input in greetings:
        msg = "👋 Hello! I'm your health assistant bot. Tell me your symptoms, and I’ll try to help."
        return jsonify({"messages": [msg]})

    if user_input in gratitude:
        msg = "😊 You're welcome! Let me know if you have any symptoms or questions."
        return jsonify({"messages": [msg]})

    if user_input in polite_words:
        msg = "👍 Got it! Please go ahead and share your symptom."
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

USER_DB = 'databases/user_db.csv'

@app.route("/login", methods=['GET', 'POST'])
def login_win():
    if request.method == 'POST':
        email = request.form.get('email')
        password  = request.form.get('password')
        
        if not os.path.exists(USER_DB):
            return render_template("login.html", message="❌ No users found.", success=False)
        
        with open(USER_DB, 'r') as f:
            all_lines = f.readlines()
            lines = all_lines[1:] if "first_name" in all_lines[0].lower() else all_lines
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                
                stored_email = parts[3]
                stored_pass = parts[5]
                
                if stored_email == email and bcrypt.checkpw(password.encode('utf-8'), stored_pass.encode('utf-8')):
                    session['user'] = {
                    'email': stored_email,
                    'first_name': parts[0],
                    'last_name': parts[1],
                    'mobile': parts[2],
                    'dob': parts[4],
                    'password': stored_pass
                    }
                    return render_template("login.html", message="✅ Login successful!", success=True, close_window=True, username=parts[0])
                
        return render_template("login.html", message="❌ Invalid credentials.", success=False)
    
    return render_template('login.html')

@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = session.get("user")
    if not user:
        return "❌ You are not logged in.", 403

    if request.method == "POST":
        # Update only fields present in the form
        for field in ['first_name', 'last_name', 'mobile', 'dob']:
            if field in request.form:
                user[field] = request.form[field]
        session['user'] = user

        # Update CSV
        lines = []
        with open("databases/user_db.csv", "r") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            parts = lines[i].strip().split(",")
            if parts[3] == user['email']:  # email is unique
                lines[i] = f"{user['first_name']},{user['last_name']},{user['mobile']},{user['email']},{user['dob']},{user['password']}\n"

        with open("databases/user_db.csv", "w") as f:
            f.writelines(lines)

        return render_template("profile.html", user=user, message="✅ Changes saved successfully!", close_window=True)

    return render_template("profile.html", user=user)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = {
            "first_name": request.form.get("first_name"),
            "last_name": request.form.get("last_name"),
            "mobile": request.form.get("mobile"),
            "email": request.form.get("email"),
            "dob": request.form.get("dob"),
            "password": request.form.get("password"),
            "confirm_password": request.form.get("confirm_password")
        }
        
        if data["password"] != data["confirm_password"]:
            return render_template("create_acc.html", message="❌ Passwords do not match.", success=False)
        
        if not data["email"] or "@" not in data["email"]:
            return render_template("register.html", message="❌ Invalid email address.", success=False)

        if not data["mobile"].isdigit() or len(data["mobile"]) != 10:
            return render_template("register.html", message="❌ Invalid mobile number.", success=False)

        # Hash the password
        hashed_pw = bcrypt.hashpw(data["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        
        with open(USER_DB, 'a') as f:
            f.write(f"{data['first_name']},{data['last_name']},{data['mobile']},{data['email']},{data['dob']},{hashed_pw}\n")
            
        return render_template("create_acc.html", message="✅ Account created successfully. You may now log in.", success=True)
        
    return render_template('create_acc.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(debug=True)
