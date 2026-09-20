import streamlit as st
import spacy
from collections import Counter
import heapq
import nltk
from nltk.stem import PorterStemmer

# Initialize NLTK Stemmer
stemmer = PorterStemmer()

# Load the NLP model once to keep the app fast
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

st.title("Complete NLP Text Analyzer")
st.markdown("Features: Tokenization, POS Tagging, NER, Dependency Parsing, Relationships, Stemming, Lemmatization, Stopwords, Word Frequency, and Summarization.")

# Create the text input box
user_text = st.text_area("Paste your text here to test the pipeline:", height=150)

# Create the button
if st.button("Process Text"):
    if user_text:
        doc = nlp(user_text)
        
        # --- 1. TOKEN ANALYSIS (Covers Tokenization, POS, Lemma, Stemming, Stopwords) ---
        st.subheader("1. Token Analysis")
        token_data = []
        for token in doc:
            token_data.append({
                "Token": token.text,
                "POS Tag": token.pos_,
                "Lemma": token.lemma_,
                "Stem": stemmer.stem(token.text),
                "Is Stopword?": token.is_stop
            })
        st.dataframe(token_data) # Displays as a neat table

        # --- 2. NAMED ENTITY RECOGNITION (NER) ---
        st.subheader("2. Named Entity Recognition (NER)")
        if doc.ents:
            ner_data = [{"Entity": ent.text, "Label": ent.label_, "Description": spacy.explain(ent.label_)} for ent in doc.ents]
            st.dataframe(ner_data)
        else:
            st.write("No named entities found in this text.")

        # --- 3. WORD FREQUENCY ---
        st.subheader("3. Word Frequency")
        # Extract words ignoring stopwords, punctuation, and whitespace
        words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        word_freq = Counter(words)
        
        if word_freq:
            freq_data = [{"Word": word, "Frequency": count} for word, count in word_freq.most_common(10)]
            st.dataframe(freq_data)
        else:
            st.write("Not enough meaningful words to calculate frequency.")

        # --- 4. RELATIONSHIP EXTRACTION (Using Dependency Parsing) ---
        st.subheader("4. Entity Relationships")
        relationships = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ != "PRON":
                subject = token.text
                verb = token.head
                for child in verb.children:
                    if child.dep_ in ("dobj", "attr"):
                        relationships.append({"Subject": subject, "Relation": verb.lemma_, "Object": child.text})
                    elif child.dep_ == "prep":
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                relationships.append({"Subject": subject, "Relation": f"{verb.lemma_} {child.text}", "Object": prep_child.text})
        
        if relationships:
            st.dataframe(relationships)
        else:
            st.write("No clear Subject-Verb-Object relationships found.")

        # --- 5. EXTRACTIVE SUMMARIZATION ---
        st.subheader("5. Extractive Summary")
        if word_freq:
            max_freq = max(word_freq.values())
            normalized_freq = {word: freq / max_freq for word, freq in word_freq.items()}
            
            sentence_scores = {}
            for sent in doc.sents:
                for word in sent:
                    if word.text.lower() in normalized_freq:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + normalized_freq[word.text.lower()]
            
            # Pick the top 2 highest scoring sentences
            summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
            
            # Reorder them based on their original appearance in the text
            summary = " ".join([sent.text.strip() for sent in doc.sents if sent in summary_sentences])
            st.write(summary)
            
    else:
        st.warning("Please enter some text first.")
