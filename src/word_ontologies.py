import streamlit as st
from components import load_model, word_analogy

def show_word_analogies():
    st.subheader("Word Analogies")
    st.markdown("Explore word analogies like `king - man + woman = queen`.")
    positive_words = st.text_input("Enter positive words (comma-separated):", value="king, woman")
    negative_words = st.text_input("Enter negative words (comma-separated):", value="man")

    if st.button("Compute Analogy"):
        model = load_model('glove-wiki-gigaword-100')
        positive_list = [w.strip() for w in positive_words.split(",")]
        negative_list = [w.strip() for w in negative_words.split(",")]
        result = word_analogy(model, positive=positive_list, negative=negative_list)
        if result:
            st.write(f"Result: **{result[0][0]}** with similarity score of **{result[0][1]:.4f}**")
        else:
            st.write("Could not compute analogy with the provided words.")
