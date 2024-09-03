import streamlit as st
from components import type_text

# Set the page configuration as the very first Streamlit command
st.set_page_config(
    page_title="Word Embedding Explorer",
    page_icon="icon.png"  # Ensure this path is correct
)

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.write('A sample text')

def show_home():
    st.markdown("<h3>Welcome to the advanced word embeddings vectorization tool!</h3>", unsafe_allow_html=True)
    type_text("Explore various features using the tabs above.", delay=0.05, placeholder=st.empty())