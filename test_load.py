

from emotion_engine.load_and_preprocess import load_goemotions_dataset

df = load_goemotions_dataset()
print(df.sample(5))
