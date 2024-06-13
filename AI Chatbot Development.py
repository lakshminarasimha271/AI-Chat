import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample corpus for the chatbot
corpus = [
    'Hello, how are you?',
    'I am good, thank you.',
    'What is your name?',
    'My name is ChatBot.',
    'How can I help you?',
    'Tell me about yourself.',
    'Exit'
]

# Initialize the tokenizer and TF-IDF vectorizer
tokenizer = nltk.tokenize.TweetTokenizer()
vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize)

# Tokenize and transform the corpus into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(corpus)

def get_response(user_input):
    # Transform user input into TF-IDF vector
    input_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(input_vector, tfidf_matrix)

    # Find the index of the most similar response
    idx = np.argmax(similarities)

    return corpus[idx]

if __name__ == "__main__":
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print("Chatbot:", response)
