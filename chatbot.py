import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Télécharger les données NLTK nécessaires
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4') 
nltk.download('stopwords')
# Prétraitement du texte
def preprocess(text):
    # Conversion en minuscules
    text = text.lower()
    # Suppression de la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Suppression des mots vides (stop words)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatisation des mots
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Fonction pour trouver la phrase la plus pertinente
def get_most_relevant_sentence(user_query, sentences):
    # Prétraite la requête de l'utilisateur et les phrases
    user_query = preprocess(user_query)
    sentences = [preprocess(sentence) for sentence in sentences]
    
    # Calcul de la similarité
    vectorizer = TfidfVectorizer().fit_transform([user_query] + sentences)
    similarity_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    most_relevant_index = similarity_scores.argmax()
    return sentences[most_relevant_index]

# Fonction principale du chatbot
def chatbot(user_query, sentences):
    # Obtenir la phrase la plus pertinente
    response = get_most_relevant_sentence(user_query, sentences)
    return response

# Fonction principale pour l'interface Streamlit
def main():
    # Charger le fichier texte et diviser en phrases
    with open("cours.txt", "r",encoding="utf-8") as file:
        text = file.read()
    sentences = text.split('.')
    
    st.title("Chatbot basé sur La Gestion de Projet")
    st.write("Posez une question à propos du Management de projet!")
    
    user_query = st.text_input("Votre question :")
    
    if user_query:
        response = chatbot(user_query, sentences)
        st.write("Réponse : ", response)

if __name__ == "__main__":
    main()

