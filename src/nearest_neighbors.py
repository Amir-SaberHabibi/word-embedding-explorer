import streamlit as st
import pandas as pd
from components import load_model, get_nearest_neighbors

def show_nearest_neighbors():
    st.subheader("Nearest Neighbors")
    st.markdown("Find the closest words to a given word in the vector space.")
    word = st.text_input("Enter a word:")

    with st.expander("Select Number of Neighbors"):
        num_neighbors = st.slider("Number of neighbors:", 3, 10, 5)
        st.markdown("""
            <style>
            .css-1d391kg {
                border: 2px solid lightgreen;
            }
            </style>
            """, unsafe_allow_html=True)
            
    if st.button("Find Nearest Neighbors"):
        model = load_model('glove-wiki-gigaword-100')
        neighbors = get_nearest_neighbors(model, word, topn=num_neighbors)
        if neighbors:
            st.write(f"Nearest neighbors to **{word}**:")
            df_neighbors = pd.DataFrame(neighbors, columns=["Neighbor", "Similarity Score"])
            st.dataframe(df_neighbors)
        else:
            st.write("Word not found in the model.")
