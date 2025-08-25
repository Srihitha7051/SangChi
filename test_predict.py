from emotion_engine.predict_emotions import predict_emotions

text = "I'm honestly so happy and thankful for your help!"
emotions = predict_emotions(text)

print("🎯 Predicted Emotions:")
for emo, score in emotions:
    print(f"  - {emo}: {score}")
