import streamlit as st
from homepage import show_home
from word_ontologies import show_word_analogies
from nearest_neighbors import show_nearest_neighbors
from word_similarity import show_word_similarity
from verctor_space import show_interactive_exploration
from word_embeddings import show_word_embeddings

def main():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Home", "Word Analogies", "Nearest Neighbors", 
        "Word Similarity", "Explore Vector Space", 
        "Word Embeddings"
    ])

    with tab1:
        show_home()

    with tab2:
        show_word_analogies()

    with tab3:
        show_nearest_neighbors()

    with tab4:
        show_word_similarity()

    with tab5:
        show_interactive_exploration()

    with tab6:
        show_word_embeddings()

if __name__ == "__main__":
    main()
