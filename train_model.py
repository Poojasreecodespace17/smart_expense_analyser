import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

df = pd.read_csv("BigBasket Products.csv")

df = df[["product", "category"]]

df = df.dropna()

df["product"] = df["product"].str.lower().str.strip()

df.columns = ["item", "category"]

df = df.drop_duplicates()

print("Dataset size:", len(df))

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["item"])
y = df["category"]


model = LinearSVC()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained using BigBasket dataset!")