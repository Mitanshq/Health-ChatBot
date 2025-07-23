from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from main import predict_disease, predict_cure, reinforcement_learning, apply_renf_learning
from difflib import get_close_matches
import threading, os, bcrypt, random, re
from sqlite3 import *

app = Flask(__name__)
app.secret_key = "secret"

CHAT_FILE = 'chats.json'

negative_phrases = [
    'no', 'nah', 'nope', 'not really', 'i am fine', 'nothing', 'np',
    'thank you', 'you are right', 'correct', 'right', 'no problem'
]

ini_prompt = [
    "Tell me your new symptoms so I can help you better!",
    "Please share your symptoms so I can assist you.",
    "What symptoms are you experiencing now?",
    "Let me know how you're feeling.",
    "Describe your current health issues.",
    "Could you tell me what you're feeling?",
    "I’m here to help—what symptoms do you have?",
    "What problems or discomfort are you facing?",
    "Share your symptoms to proceed.",
    "Tell me what’s bothering you.",
    "Let’s start again—what symptoms are present?",
    "Can you list the symptoms you're feeling now?",
    "I need your symptoms to guide you properly.",
    "What signs or symptoms are you noticing?",
    "Mention the issues you're dealing with currently.",
    "Input your symptoms to get accurate suggestions."
]

more_prompt = [
    "Tell me your new symptoms so I can help you better!",
    "Please tell me what other symptoms you're experiencing.",
    "Can you describe any other issues you're facing?",
    "Share more symptoms so I can assist you properly.",
    "What else are you feeling?",
    "Let me know any other symptoms you're noticing.",
    "I'd like to know more about how you're feeling.",
    "Tell me about any additional discomfort you're having.",
    "Are there other symptoms you’d like to mention?",
    "Please list all the symptoms you have.",
    "What other signs or problems are you experiencing?",
    "Is there anything else unusual you're feeling?",
    "Add more symptoms if you're feeling anything else.",
    "What additional health issues are you noticing?",
    "I need more details — tell me more symptoms."
]

not_confident_phrases = [
    "I'm not fully confident.",
    "I'm a bit unsure at this point.",
    "I need more information to be certain.",
    "I’m not entirely sure yet.",
    "It’s hard to tell right now.",
    "I can't say for sure yet.",
    "My prediction isn't strong enough.",
    "I'm having trouble making a clear diagnosis.",
    "The symptoms aren't specific enough yet.",
    "I’m still unsure based on the given info.",
    "It’s not clear to me just yet.",
    "I'm detecting a few possibilities, not one clear answer.",
    "I need a few more clues to be accurate.",
    "It’s difficult to be confident right now.",
    "I might be missing something — help me with more details."
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

possible_conditions_phrases = [
    "You might be experiencing one of the following:",
    "Here are a few conditions that match your symptoms:",
    "These could be the possible diagnoses:",
    "Based on your input, these are some likely conditions:",
    "I suspect one of the following conditions:",
    "It might be related to one of these:",
    "Here are a few possibilities:",
    "These are some conditions that could be causing your symptoms:",
    "You may be dealing with one of these:",
    "These conditions match your symptoms closely:",
    "This could point to any of the following:",
    "Several conditions seem to fit. They include:",
    "Your symptoms align with these potential conditions:",
    "It’s possibly one of these:",
    "These are the most probable causes I found:"
]

predicted_disease_phrases = [
    "✅ It seems like you may have",
    "✅ Based on the symptoms, you could have",
    "✅ You might be showing signs of",
    "✅ My analysis suggests it could be",
    "✅ This may be a case of",
    "✅ Possibly, you’re experiencing",
    "✅ I suspect you might have",
    "✅ These symptoms often relate to",
    "✅ Looks like it could be",
    "✅ There’s a good chance this is",
    "✅ You may be affected by",
    "✅ I’d say it appears to be",
    "✅ Your symptoms match with",
    "✅ Most likely, this is",
    "✅ I'm fairly confident this is"
]

medication_phrases = [
    "💊 Here are the suggested medications:",
    "💊 You may consider taking the following medicines:",
    "💊 Recommended drugs for this condition include:",
    "💊 Possible treatment options are:",
    "💊 Based on your symptoms, try these medications:",
    "💊 Suggested pharmaceutical aids:",
    "💊 The following medicines might help:",
    "💊 Medications you can take:",
    "💊 Here’s what you can use to treat it:",
    "💊 Listed medications are typically used:",
    "💊 You could benefit from these treatments:",
    "💊 Common prescriptions include:",
    "💊 Try the following drugs:",
    "💊 Doctors usually recommend:",
    "💊 Consider these for relief:"
]

remedy_phrases = [
    "🏡 Here are some helpful home remedies:",
    "🏡 You can try the following natural treatments:",
    "🏡 Suggested remedies you can do at home:",
    "🏡 These home treatments might ease your symptoms:",
    "🏡 Try these at-home solutions:",
    "🏡 Natural ways to feel better include:",
    "🏡 Home-based remedies to consider:",
    "🏡 You might find relief with these home cures:",
    "🏡 Consider these traditional remedies:",
    "🏡 Safe and easy remedies to try at home:",
    "🏡 Here’s how you can treat it naturally:",
    "🏡 Use these home remedies for relief:",
    "🏡 Simple home tips that can help:",
    "🏡 Household treatments you can follow:",
    "🏡 These might help if you're treating it at home:"
]

feedback_phrases = [
    "🤔 Did that help you?",
    "✅ Was this information useful to you?",
    "💬 Does this answer your concern?",
    "🧐 Did I get that right?",
    "❓Was this what you were looking for?",
    "👍 Was that helpful?",
    "✅ Do you feel this solved your issue?",
    "🙂 Was this suggestion useful?",
    "📩 Did this advice help you?",
    "🙋‍♂️ Was that the info you needed?",
    "✅ Does this make sense to you?",
    "🤖 Was this accurate for you?",
    "🙌 Was this response okay?",
    "🧠 Did this help you understand better?",
    "🔍 Did that clear things up?"
]

invalid_symptom_phrases = [
    "❌ I couldn't recognize that as a symptom. Please try again.",
    "❌ That doesn’t look like a health issue. Can you rephrase?",
    "❌ Hmm... That doesn’t seem to be a valid medical symptom.",
    "❌ I’m not familiar with that. Try entering a real symptom.",
    "❌ That input doesn’t match any known symptoms. Please check it.",
    "❌ I couldn’t find any health condition related to that.",
    "❌ That doesn't appear to be something I can diagnose. Try another symptom.",
    "❌ Please enter an actual symptom for me to help you.",
    "❌ I'm not trained to understand that input as a symptom.",
    "❌ That doesn’t seem health-related. Please enter a symptom.",
    "❌ Try again with a medically relevant issue.",
    "❌ That doesn’t seem like something I can assess. Try a real symptom.",
    "❌ It looks like you entered something unrelated. Please try a valid symptom.",
    "❌ Sorry, but I can only help with known health symptoms.",
    "❌ That term isn’t recognized as a symptom. Can you rephrase it?"
]

new_session_phrases = [
    "🔄 You can begin a new session by sharing your symptoms or asking about a disease.",
    "🆕 Let's start fresh! Enter your symptoms or mention a condition you're curious about.",
    "💬 Ready when you are — share your symptoms or ask about any illness.",
    "🩺 Please go ahead and tell me your symptoms or name a disease to explore.",
    "📋 You can now type new symptoms or ask for info on a health condition.",
    "🌐 Restarting... enter symptoms or inquire about any disease.",
    "🔁 Start a new chat by telling me how you're feeling or what you want to know.",
    "🗣️ Share your symptoms again or ask about another health issue.",
    "📤 New session started. Type your symptoms or ask me about a disease.",
    "💡 Feel free to describe new symptoms or get details about any condition.",
    "🔍 Let's begin again! Tell me your symptoms or the disease you're concerned about.",
    "🎯 You may now enter a new symptom list or ask about a different disease.",
    "⚕️ I'm ready — tell me how you're feeling or what you'd like to know.",
    "📣 Go ahead and enter symptoms or ask about another diagnosis.",
    "🩹 Begin your health check again — symptoms or disease name, please!"
]

all_known_symptoms = set()
for symptoms in known_diseases.values():
    all_known_symptoms.update(s.strip().lower() for s in symptoms)

def disease_to_sypmtom(disease):
    return known_diseases.get(disease.lower())

@app.route('/')
def home():
    session['symptoms'] = []
    session['steps'] = 0
    return render_template('index.html', user=session.get("user"))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "").strip().lower()

    def clean_text(text):
        cleaned = text.lower().strip()
        cleaned = re.sub(
            r'\b(a|lot of|lots of|too much|very|mild|severe|kind of|bit of|some|can|you|describe|tell|me|more|about|the|called|info|disease|mention|any|information|in|detail|details|brief|briefly|us|will|educate|bout|less|bit|much|high|highly|lessly)\b',
            '', cleaned)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # removes all punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
        
    cleaned_user_ip = clean_text(user_input)
        
    def validate_symptom(symptom):
        symptom = clean_text(symptom)
        if symptom in all_known_symptoms:
            return symptom
        match = get_close_matches(symptom, all_known_symptoms, n=1, cutoff=0.85)
        return match[0] if match else None

    # Check if it's a known disease query
    if cleaned_user_ip in known_diseases:
        symptoms = disease_to_sypmtom(cleaned_user_ip)
        cure = predict_cure(cleaned_user_ip)
        messages = [
            f"You mentioned {cleaned_user_ip}",
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
            reinforcement_learning(predicted_data['symptoms'], predicted_data['disease'], cleaned_user_ip)
            threading.Thread(target=apply_renf_learning, daemon=True).start()
            session.pop("last_prediction", None)
            new_sess_prompt = random.choice(new_session_phrases)
            messages = [
                "✅ Feedback saved.",
                f'{new_sess_prompt}'
            ]
            return jsonify({"messages": messages})

    # Reset on negatives
    if user_input in negative_phrases:
        session['symptoms'] = []
        session['steps'] = 0
        messages = [
            "🟢 Session reset.",
            random.choice(ini_prompt)
        ]
        return jsonify({"messages": messages})

    # Small talk handling
    greetings = ["hi", "hello", "hey", "good morning", "good evening", 'gm', 'wassup', 'hey there']
    gratitude = ["thanks", "thank you", "thnx", "thank u", "ty", 'tysm']
    polite_words = ["please", "okay", "ok", "hmm", "cool", 'pls', 'plss']
    greeting_phrases = [
    "👋 Hi there! I'm your health assistant. What symptoms are you experiencing?",
    "🤖 Hello! I’m here to help you with health issues. Tell me your symptoms.",
    "🩺 Hey! I’m your medical assistant. Describe what you're feeling.",
    "🙌 Welcome! Please share any symptoms so I can assist you.",
    "👋 Hi! I'm here to guide you through any health concerns. Let’s start with your symptoms.",
    "👨‍⚕️ Hello! I can help diagnose based on your symptoms. What are you feeling?",
    "🧠 Hi! I’m trained to assist with health info. Tell me what’s bothering you.",
    "🖐️ Greetings! I’m here to support you. What symptoms do you have?",
    "💬 Hello there! Share your health concerns or symptoms, and I’ll do my best to help.",
    "👋 Welcome! You can start by describing what you’re feeling.",
    "🤝 Hello! Let’s figure out what’s going on. What symptoms are you facing?",
    "🌟 Hi! Share how you're feeling and I’ll try to identify possible conditions.",
    "🧑‍⚕️ Hey there! Tell me your symptoms so I can assist with some advice.",
    "📋 Hello! Start by listing any symptoms you're experiencing.",
    "🌈 Hi! I’m your friendly health assistant bot. What’s troubling you today?"
]
    gratitude_responses = [
    "😊 You're welcome! I'm here if you have more symptoms or queries.",
    "😄 Anytime! Feel free to ask anything health-related.",
    "🤗 No problem at all! Let me know if you’re feeling anything unusual.",
    "🙌 Glad I could help! Got more symptoms or questions?",
    "🙂 You’re very welcome! I’m ready for your next symptom or doubt.",
    "💬 Sure thing! Tell me if anything else is bothering you.",
    "😌 Happy to help! What else can I assist you with?",
    "😁 You’re welcome! Want to talk about another symptom?",
    "🩺 No worries! I’m here for your health questions.",
    "🙏 Of course! Do share if anything else is troubling you.",
    "👩‍⚕️ Always here to help! Got another concern?",
    "👍 You got it! Let me know if you feel anything odd.",
    "🥼 Anytime! I'm listening — what symptoms do you have?",
    "🤖 My pleasure! Got any more health concerns?",
    "✅ You're welcome! Let me know if you’d like to continue."
]
    polite_response_phrases = [
    "👍 Got it! Please go ahead and share your symptom.",
    "👌 Understood! Let me know what you're experiencing.",
    "📝 Alright! Tell me your symptom so I can assist.",
    "✅ Okay! What symptom would you like to share?",
    "🫡 Sure! Please describe how you're feeling.",
    "🗣️ Alrighty! Share your symptom when you're ready.",
    "📋 Got you! Go ahead and tell me your health issue.",
    "👍 Perfect! Now just tell me your symptom.",
    "🙂 Noted! What are you feeling right now?",
    "🩺 Thanks! Please share your symptoms with me.",
    "💡 Okay, great! What problem can I help with?",
    "👂 I’m listening! What symptom are you having?",
    "🫶 Got it. Let me know what’s troubling you.",
    "📢 Go ahead! I'm ready for your symptom.",
    "🚨 Sure! Describe the health concern you're facing."
]


    if user_input in greetings:
        greeting_prompt = random.choice(greeting_phrases)
        return jsonify({"messages": [f'{greeting_prompt}']})
    if user_input in gratitude:
        grat_prompt = random.choice(gratitude_responses)
        return jsonify({"messages": [f'{grat_prompt}']})
    if user_input in polite_words:
        polite_prompt = random.choice(polite_response_phrases)
        return jsonify({"messages": [f'{polite_prompt}']})

    # Validate input symptom
    validated = validate_symptom(cleaned_user_ip)
    if not validated:
        invalid_prompt = random.choice(invalid_symptom_phrases)
        return jsonify({"messages": [f'{invalid_prompt}']})

    # Accumulate symptom
    symptoms = session.get('symptoms', [])
    symptoms.append(validated)
    session['symptoms'] = symptoms
    session['steps'] = session.get('steps', 0) + 1

    # Predict
    response = predict_disease(", ".join(symptoms))

    if response['status'] == 'ambiguous':
        messages = [
            f"{random.choice(possible_conditions_phrases)} {', '.join([d for d, _ in response['possible_diseases']])}.",
            f"{random.choice(not_confident_phrases)}",
            f"{random.choice(more_prompt)}"
        ]
        return jsonify({"messages": messages})

    elif response['status'] == 'confident':
        disease = response['predicted']
        cure = predict_cure(disease)
        session['last_prediction'] = {
            "symptoms": symptoms.copy(),
            "disease": disease
        }
        session['steps'] = 0
        session['symptoms'] = []
        messages = [
            f"{random.choice(predicted_disease_phrases)} {disease}",
            f"{random.choice(medication_phrases)} {', '.join(set(cure['medication']))}",
            f"{random.choice(remedy_phrases)} {', '.join(set(cure['remedies']))}",
            f"{random.choice(feedback_phrases)} (YES/NO)"
        ]
        return jsonify({"messages": messages})

    return jsonify({"messages": ["❌ Sorry, something went wrong. Please try again."]})

USER_DB = 'databases/user_db.csv'

os.makedirs(os.path.dirname(USER_DB), exist_ok=True)

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
        
        os.makedirs(os.path.dirname(USER_DB), exist_ok=True)

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

        os.makedirs(os.path.dirname(USER_DB), exist_ok=True)
        
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
