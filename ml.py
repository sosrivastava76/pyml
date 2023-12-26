import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Example data, replace with your dataset
data = [
    ("I love this product! It's amazing.", "positive"),
    ("This movie is terrible.", "negative"),
    # Add more examples
]

# Separate text and labels
texts, labels = zip(*data)

# Tokenization and removal of stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Apply preprocessing to all texts
texts = [preprocess_text(text) for text in texts]
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


new_text = "I feel great today!"
new_text = preprocess_text(new_text)
new_text_features = tfidf_vectorizer.transform([new_text])

prediction = classifier.predict(new_text_features)
print(f"The predicted mood is: {prediction[0]}")

