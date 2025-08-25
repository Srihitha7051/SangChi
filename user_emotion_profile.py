from emotion_engine.predict_emotions import predict_emotions
from collections import Counter
import numpy as np

def get_user_emotion_vector(answers, all_emotions):
    """
    Predicts emotions from a user's list of text answers and returns a vector.
    - answers: List of strings (user's answers)
    - all_emotions: List of all emotion labels (binarizer.classes_)
    """
    emotion_counter = Counter()

    for ans in answers:
        emotions = predict_emotions(ans)
        for emo, _ in emotions:
            emotion_counter[emo] += 1

    total = sum(emotion_counter.values()) or 1

    # Normalize counts into vector
    vector = [emotion_counter[emo] / total for emo in all_emotions]
    return np.array(vector)

if __name__ == "__main__":
    sample_answers = [
        "I feel overwhelmed with everything going on.",
        "Honestly, I'm so excited about tomorrow!",
        "I'm grateful for my friends who always support me.",
        "It really hurt when they ignored me.",
        "I'm just confused about what I should do.",
        "Sometimes I feel nothing at all.",
        "I want to scream from frustration!",
        "I laughed so much today!",
        "Why did this happen to me?",
        "Thanks for being there when I needed someone."
    ]

    # Load emotion labels
    import joblib
    binarizer = joblib.load("model/emotion_binarizer.pkl")
    all_emotions = binarizer.classes_

    vector = get_user_emotion_vector(sample_answers, all_emotions)

    print("🎯 Emotion Vector:")
    print(vector)
    print("✅ Script executed.")
