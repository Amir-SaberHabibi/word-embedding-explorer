import streamlit as st
from components import type_text



with st.sidebar:
    st.write('A sample text')

def show_home():
    st.markdown("<h3>Welcome to the advanced word embeddings vectorization tool!</h3>", unsafe_allow_html=True)
    type_text("Explore various features using the tabs above.", delay=0.05, placeholder=st.empty())