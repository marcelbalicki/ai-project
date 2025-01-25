import streamlit as st
from transformers import pipeline

st.title("📊 Analizator Emocji")
user_input = st.text_area("Wpisz komentarz:")

if user_input:
    classifier = pipeline("text-classification", 
                         model="j-hartmann/emotion-english-distilroberta-base")
    result = classifier(user_input)[0]
    
    st.subheader("Wynik analizy:")
    st.metric(label="Dominująca emocja", 
             value=result['label'].upper(), 
             delta=f"{result['score']*100:.1f}% pewności")
    
    st.write("**Pełny rozkład emocji:**")
    st.bar_chart({entry['label']: entry['score'] 
                for entry in classifier(user_input, return_all_scores=True)[0]})
