import streamlit as st
from components import load_model, word_similarity

def show_word_similarity():
    st.subheader("Word Similarity")
    st.markdown("Compute and display the similarity score between two words.")
    word1 = st.text_input("Enter the first word:")
    word2 = st.text_input("Enter the second word:")

    if st.button("Compute Similarity"):
        model = load_model('glove-wiki-gigaword-100')
        similarity = word_similarity(model, word1, word2)
        if similarity is not None:
            st.write(f"Similarity between **{word1}** and **{word2}**: **{similarity:.4f}**")
        else:
            st.write("One or both words not found in the model.")