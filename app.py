from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from user_emotion_profile import get_user_emotion_vector
import joblib
from datetime import datetime
from collections import defaultdict
from flask import session
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_migrate import Migrate


app = Flask(__name__)
app.secret_key = 'super-secret-key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sanchi.db'  # this creates a file called sanchi.db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load model once globally
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")






# Load emotion binarizer (for consistent vector size)
binarizer = joblib.load("model/emotion_binarizer.pkl")
all_emotions = binarizer.classes_






# Convert answer list to a single embedding
def get_user_embedding(answers):
    if not answers:
        return None
    texts = [a.answer for a in answers]
    combined = " ".join(texts)  # Combine all answers
    inputs = tokenizer(combined, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(100), nullable=True)

class UserAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    question_number = db.Column(db.Integer, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Thought model
class Thought(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), db.ForeignKey('user.email'), nullable=False)
    thought = db.Column(db.Text, nullable=False)
    mood = db.Column(db.String(50), nullable=True)
    is_public = db.Column(db.Boolean, default=True)
    public = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.String(100))

class MatchResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    matched_email = db.Column(db.String(120), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    top_emotions = db.Column(db.String(250))  # comma-separated string of matching emotions
    explanation = db.Column(db.Text, nullable=True)



DATA_FILE = 'users.json'

MESSAGES_FILE = 'messages.json'

if os.path.exists(MESSAGES_FILE):
    with open(MESSAGES_FILE, 'r') as f:
        messages = json.load(f)
else:
    messages = []

# Load existing users from JSON file if it exists
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        users = json.load(f)
else:
    users = []

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/profile', methods=['GET', 'POST'])
# def profile():
#     if 'email' not in session:
#         return redirect('/login')

#     user = next((u for u in users if u['email'] == session['email']), None)
#     if request.method == 'POST':
#         user['thoughts'] = request.form['thoughts']
#         user['feelings'] = request.form['feelings']
#         user['interests'] = request.form['interests']
#         user['habits'] = request.form['habits']
#         with open(DATA_FILE, 'w') as f:
#             json.dump(users, f, indent=4)
#         return redirect('/matches')
#     return render_template('profile.html', user=user)

@app.route('/q1', methods=['GET', 'POST'])
def question1():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=1,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q2')  # Go to next question

    return render_template('q1.html')

@app.route('/q2', methods=['GET', 'POST'])
def question2():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=2,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q3')  # Go to next question

    return render_template('q2.html')

@app.route('/q3', methods=['GET', 'POST'])
def question3():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=3,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q4')  # Go to next question

    return render_template('q3.html')

@app.route('/q4', methods=['GET', 'POST'])
def question4():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=4,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q5')  # Go to next question

    return render_template('q4.html')

@app.route('/q5', methods=['GET', 'POST'])
def question5():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=5,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q6')  # Go to next question

    return render_template('q5.html')

@app.route('/q6', methods=['GET', 'POST'])
def question6():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=6,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q7')  # Go to next question

    return render_template('q6.html')

@app.route('/q7', methods=['GET', 'POST'])
def question7():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=7,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q8')  # Go to next question

    return render_template('q7.html')

@app.route('/q8', methods=['GET', 'POST'])
def question8():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=8,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q9')  # Go to next question

    return render_template('q8.html')

@app.route('/q9', methods=['GET', 'POST'])
def question9():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=9,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/q10')  # Go to next question

    return render_template('q9.html')

@app.route('/q10', methods=['GET', 'POST'])
def question10():
    if 'email' not in session:
        return redirect('/login')

    if request.method == 'POST':
        answer = request.form['answer']
        new_answer = UserAnswer(
            email=session['email'],
            question_number=10,
            answer=answer
        )
        db.session.add(new_answer)
        db.session.commit()
        return redirect('/matches')  # Go to next question

    return render_template('q10.html')


# @app.route('/matches')
# def matches():
#     if 'email' not in session:
#         return redirect('/login')

#     current_user = next((u for u in users if u['email'] == session['email']), None)
#     if not current_user:
#         return "User not found."

#     other_users = [u for u in users if u['email'] != current_user['email']]
#     if not other_users:
#         return "No matches yet."

#     combined_profiles = [
#         " ".join([u['thoughts'], u['feelings'], u['interests'], u['habits']])
#         for u in other_users
#     ]
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(combined_profiles + [
#         " ".join([current_user['thoughts'], current_user['feelings'], current_user['interests'], current_user['habits']])
#     ])
#     similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
#     best_match_index = similarity_scores.argmax()
#     top_match = other_users[best_match_index]

#     return render_template('matches.html', user=current_user, matches=[top_match])


# @app.route('/matches')
# def matches():
#     current_email = session.get('email')
#     if not current_email:
#         return redirect('/login')
    
#     user = User.query.filter_by(email=current_email).first()
#     if not user:
#         # handle case when user not found in DB
#         return "User not found", 404

#     # Fetch current user's answers
#     current_user_answers = UserAnswer.query.filter_by(email=current_email).all()
    
#     # Fetch all other users' answers
#     other_users_answers = UserAnswer.query.filter(UserAnswer.email != current_email).all()
    
#     # Convert to dict for easier processing
#     def build_answer_dict(answers):
#         d = {}
#         for ans in answers:
#             d[ans.question_number] = ans.answer
#         return d
    
#     current_answers_dict = build_answer_dict(current_user_answers)
    
#     # Example: simple matching logic (customize as needed)
#     matched_users = {}
#     for ans in other_users_answers:
#         if ans.email not in matched_users:
#             matched_users[ans.email] = []
#         # Compare question answers here and collect matching score or criteria
#         if current_answers_dict.get(ans.question_number) == ans.answer:
#             matched_users[ans.email].append(ans.question_number)
    
#     # Filter users with some minimum matches
#     final_matches = [email for email, matched_qs in matched_users.items() if len(matched_qs) >= 3]  # e.g., 3 matches min
    
#     return render_template('matches.html', matches=final_matches)





@app.route('/matches')
def matches_loader():
    return render_template('finding_match.html')

@app.route('/matches/result')
def matches_result():
    current_user_email = session.get('email')
    if not current_user_email:
        return redirect(url_for('login'))

    # Get current user's answers
    current_user_answers = UserAnswer.query.filter_by(email=current_user_email).order_by(UserAnswer.question_number).all()
    if not current_user_answers:
        return "You have not answered any questions yet."

    current_texts = [ans.answer for ans in current_user_answers]
    current_vec = get_user_emotion_vector(current_texts, all_emotions)

    # Get all other users' answers
    all_other_answers = UserAnswer.query.filter(UserAnswer.email != current_user_email).all()

    users_answers = defaultdict(list)
    for ans in all_other_answers:
        users_answers[ans.email].append(ans)

    top_match_email = None
    top_score = -1

    for email, answers in users_answers.items():
        texts = [ans.answer for ans in answers]
        vec = get_user_emotion_vector(texts, all_emotions)

        if vec is not None and np.any(vec):
            score = cosine_similarity([current_vec], [vec])[0][0]
            if score > top_score:
                top_score = score
                top_match_email = email

    if not top_match_email:
        return render_template('matches.html', match=None)

    matched_user = User.query.filter_by(email=top_match_email).first()
    match_data = {
        "name": matched_user.name if matched_user else "Unknown",
        "email": top_match_email,
        "similarity": int(top_score * 100)
    }
    new_match = MatchResult(
    user_email=current_user_email,
    matched_email=top_match_email,
    similarity_score=0.78,
    explanation="Both users showed high joy and empathy in their answers."
    )
    db.session.add(new_match)
    db.session.commit()

    return render_template('matches.html', match=match_data)


@app.route('/matches/history')
def match_history():
    current_user_email = session.get('email')
    if not current_user_email:
        return redirect(url_for('login'))

    history = MatchResult.query.filter_by(user_email=current_user_email).order_by(MatchResult.id.desc()).all()

    return render_template('match_history.html', matches=history)


# @app.route('/matches/result')
# def matches_result():
#     current_user_email = session.get('email')
#     if not current_user_email:
#         return redirect(url_for('login'))

#     current_user_answers = UserAnswer.query.filter_by(email=current_user_email).order_by(UserAnswer.question_number).all()
#     if not current_user_answers:
#         return "You have not answered any questions yet."

#     # Get embedding for current user
#     current_vec = get_user_embedding(current_user_answers)
#     if current_vec is None:
#         return "Insufficient data."

#     # Collect all other users' vectors
#     all_other_answers = UserAnswer.query.filter(UserAnswer.email != current_user_email).all()

#     users_answers = defaultdict(list)
#     for ans in all_other_answers:
#         users_answers[ans.email].append(ans)

#     top_match_email, top_score = None, -1
#     for email, answers in users_answers.items():
#         vec = get_user_embedding(answers)
#         if vec is not None:
#             score = cosine_similarity([current_vec], [vec])[0][0]
#             if score > top_score:
#                 top_score = score
#                 top_match_email = email

#     if not top_match_email:
#         return render_template('matches.html', match=None)

#     matched_user = User.query.filter_by(email=top_match_email).first()
#     match_data = {
#         "name": matched_user.name if matched_user else "Unknown",
#         "email": top_match_email,
#         "similarity": int(top_score * 100)
#     }

#     return render_template('matches.html', match=match_data)






def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/chat/<email>', methods=['GET', 'POST'])
def chat(email):
    if 'email' not in session:
        return redirect('/login')
    
    sender = session['email']
    receiver = email

    # Handle new message
    if request.method == 'POST':
        text = request.form['message']
        new_msg = {
            'from': sender,
            'to': receiver,
            'message': text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        messages.append(new_msg)
        with open(MESSAGES_FILE, 'w') as f:
            json.dump(messages, f, indent=4)

    # Show conversation between the two users
    convo = [m for m in messages if (m['from'] == sender and m['to'] == receiver) or (m['from'] == receiver and m['to'] == sender)]
    
    return render_template('chat.html', messages=convo, receiver_email=receiver)


    
@app.route('/express', methods=['GET', 'POST'])
def express():
    if 'email' not in session:
        return redirect('/login')

    success = False
    if request.method == 'POST':
        email = session['email']
        thought = request.form['thought']
        mood = request.form.get('mood', '')
        is_public = request.form.get('public') == 'true'

        new_entry = Thought(
            email=email,
            thought=thought,
            mood=mood,
            public=is_public,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M')
        )

        db.session.add(new_entry)
        db.session.commit()
        success = True

    return render_template('express.html', success=success)

@app.route('/tag-select', methods=['GET', 'POST'])
def tag_select():
    if request.method == 'POST':
        selected_tags = request.form.getlist('tags')
        session['tags'] = selected_tags
        return redirect('/q1')  # or wherever your question flow starts

    return render_template('tag-select.html')


# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = hash_password(request.form['password'])
#         name = request.form['name']

#         for u in users:
#             if 'email' in u and u['email'] == email:
#                 return "Email already registered."

#         new_user = {
#             'name': name,
#             'email': email,
#             'password': password,
#             'thoughts': '',
#             'feelings': '',
#             'interests': '',
#             'habits': ''
#         }
#         users.append(new_user)
#         with open(DATA_FILE, 'w') as f:
#             json.dump(users, f, indent=4)
#         session['email'] = email
#         return redirect('/tag-select')
    
#     return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = hash_password(request.form['password'])
        name = request.form['name']

        # Check in the database instead of users.json
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "Email already registered."

        # Create new user and save to database
        new_user = User(email=email, password=password, name=name)
        db.session.add(new_user)
        db.session.commit()

        session['email'] = email
        return redirect('/tag-select')

    return render_template('signup.html')
    

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = hash_password(request.form['password'])

#         for u in users:
#             if u['email'] == email and u['password'] == password:
#                 session['email'] = email
#                 return redirect('/tag-select')
#         return "Invalid credentials."
#     return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hash_password(request.form['password'])

        # Check from the database now
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['email'] = email
            session['user_id'] = user.id  # 👈 ADD THIS LINE
            return redirect('/tag-select')
        return "Invalid credentials."
    
    return render_template('login.html')


@app.route('/my_profile')
def my_profile():
    user_email = session.get('email')
    if not user_email:
        return redirect(url_for('login'))

    user = User.query.filter_by(email=user_email).first()
    if not user:
        return "User not found"

    user_answers = UserAnswer.query.filter_by(email=user_email).order_by(UserAnswer.question_number).all()

    return render_template('my_profile.html', user=user, answers=user_answers)


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')



@app.route('/q1', methods=['GET', 'POST'])
def q1():
    if request.method == 'POST':
        user_answers['q1'] = request.form.get('answer')
        return redirect(url_for('q2'))
    return render_template('q1.html')

@app.route('/q2', methods=['GET', 'POST'])
def q2():
    if request.method == 'POST':
        user_answers['q2'] = request.form.get('answer')
        return redirect(url_for('q3'))
    return render_template('q2.html')

@app.route('/q3', methods=['GET', 'POST'])
def q3():
    if request.method == 'POST':
        user_answers['q3'] = request.form.get('answer')
        return redirect(url_for('q4'))
    return render_template('q3.html')

@app.route('/q4', methods=['GET', 'POST'])
def q4():
    if request.method == 'POST':
        user_answers['q4'] = request.form.get('answer')
        return redirect(url_for('q5'))
    return render_template('q4.html')

@app.route('/q5', methods=['GET', 'POST'])
def q5():
    if request.method == 'POST':
        user_answers['q5'] = request.form.get('answer')
        return redirect(url_for('q6'))
    return render_template('q5.html')

@app.route('/q6', methods=['GET', 'POST'])
def q6():
    if request.method == 'POST':
        user_answers['q6'] = request.form.get('answer')
        return redirect(url_for('q7'))
    return render_template('q6.html')

@app.route('/q7', methods=['GET', 'POST'])
def q7():
    if request.method == 'POST':
        user_answers['q7'] = request.form.get('answer')
        return redirect(url_for('q8'))
    return render_template('q7.html')

@app.route('/q8', methods=['GET', 'POST'])
def q8():
    if request.method == 'POST':
        user_answers['q8'] = request.form.get('answer')
        return redirect(url_for('q9'))
    return render_template('q8.html')

@app.route('/q9', methods=['GET', 'POST'])
def q9():
    if request.method == 'POST':
        user_answers['q9'] = request.form.get('answer')
        return redirect(url_for('q10'))
    return render_template('q9.html')

@app.route('/q10', methods=['GET', 'POST'])
def q10():
    if request.method == 'POST':
        user_answers['q10'] = request.form.get('answer')
        # Later we can save all this to the database here
        return "Thank you for submitting your answers!"
    return render_template('q10.html')



if __name__ == '__main__':
    app.run(debug=True)