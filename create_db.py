from app import app, db

with app.app_context():
    db.create_all()
    print("✅ user_answer table created!")
