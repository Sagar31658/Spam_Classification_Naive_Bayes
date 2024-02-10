import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('Youtube03-LMFAO.csv')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text):
    # Removing HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Removing non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenizing the text
    tokens = word_tokenize(text)
    # Lemmatization and removing stopwords
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stopwords.words('english')]
    return " ".join(tokens)

comments = [
    "I really enjoyed the video, especially the part where you explained the editing techniques.",
    "This tutorial was so helpful! I've learned a lot about video production.",
    "Can you do a follow-up on this topic? I have some questions about the software you used.",
    "Great content as always! Your explanations are always clear and easy to understand.",
    "WIN A FREE iPhone! Click this link now to enter the giveaway!!!",
    "Follow me on instagram!"
]
 

# Apply cleaning and tokenizing to the dataset
data['CLEANED_CONTENT'] = data['CONTENT'].apply(clean_and_tokenize)

# Using TfidfVectorizer with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(data['CLEANED_CONTENT'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['CLASS'], test_size=0.25, random_state=42)

# Naive Bayes classifier
classifier = MultinomialNB()

# Hyperparameter tuning using Grid Search
param_grid = {'alpha': [0.1, 0.5, 1.0, 10.0]}
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best estimator found by grid search
best_classifier = grid_search.best_estimator_

# Cross-validation on the training data
cross_val_scores = cross_val_score(best_classifier, X_train, y_train, cv=5)
mean_cross_val_score = cross_val_scores.mean()

# Predicting on training and test datasets
y_pred_train = best_classifier.predict(X_train)
y_pred_test = best_classifier.predict(X_test)

# Calculating accuracy for both training and test datasets
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("Mean Cross-Validation Score:", mean_cross_val_score)
print("Training Accuracy:", accuracy_train)
print("Test Accuracy:", accuracy_test)
print("Test Confusion Matrix:\n", conf_matrix_test)

cleaned_comments = [clean_and_tokenize(comment) for comment in comments]

# Transform the new comments using the TfidfVectorizer
transformed_comments = vectorizer.transform(cleaned_comments)

# Predicting the class of new comments
predicted_classes = best_classifier.predict(transformed_comments)

# Print predicted classes for the comments
for comment, prediction in zip(comments, predicted_classes):
    print(f"Comment: {comment}\nPredicted Class: {'Spam' if prediction == 1 else 'Not Spam'}\n")
