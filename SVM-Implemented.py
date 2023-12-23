import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
data = pd.read_csv('C:/Users/walim/Documents/Projects/Python/SA/data.csv')

train_data, test_data, train_labels, test_labels = train_test_split(
    data['Sentence'],
    data['Sentiment'],
    test_size=0.2,
    random_state=42
)

class SVM:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        y = np.where(y == 'negative', -1,1)
        X = np.c_[X, np.ones(X.shape[0])]

        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            # Calculate the hinge loss
            loss = 1 - y * np.dot(X, self.weights)
            misclassified = np.where(loss > 0)[0]
            if len(misclassified) > 0:
                # Choose a random misclassified sample
                random_index = np.random.choice(misclassified)
                
                # Update weights using the chosen sample
                self.weights += self.learning_rate * y[random_index] * X[random_index]

    def predict(self, X):
        # Add a bias term to X
        X = np.c_[X, np.ones(X.shape[0])]
        return np.where(np.dot(X, self.weights) > 0, 'positive', 'negative')

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        total = len(y)
        accuracy = correct / total
        return accuracy

    def predict_label(self, input_text, vectorizer):
        # Vectorize the input text using the provided vectorizer
        vectorized_text = vectorizer.transform([input_text]).toarray()

        # Predict the label
        prediction = self.predict(vectorized_text)

        return prediction[0]

# Vectorize text data using a count vectorizer
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data).toarray()
test_vectors = vectorizer.transform(test_data).toarray()

# Train the SVM model
svm_classifier = SVM()
svm_classifier.fit(train_vectors, train_labels)

test_accuracy = svm_classifier.accuracy(test_vectors, test_labels)
print("Test Accuracy:", test_accuracy)

input_text = "This is a positive example."
predicted_label = svm_classifier.predict_label(input_text, vectorizer)
print("Predicted Label:", predicted_label)