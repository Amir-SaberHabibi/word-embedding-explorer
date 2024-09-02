import streamlit as st
import numpy as np
import plotly.graph_objects as go
from components import load_model, compute_embeddings_and_pca

def show_interactive_exploration():
    st.subheader("Interactive Exploration")
    st.markdown("Interactively explore the vector space by selecting or hovering over words.")
    words_input = st.text_input("Enter words (comma-separated):", value="king, queen, man, woman")

    with st.expander("Select Number of Words to Display"):
        num_words = st.slider("Number of words to display:", 1, 20, 5)
        st.markdown("""
            <style>
            .css-1d391kg {
                border: 2px solid lightgreen;
            }
            </style>
            """, unsafe_allow_html=True)

    if words_input:
        model = load_model('glove-wiki-gigaword-100')
        words = [w.strip() for w in words_input.split(",") if w.strip() in model]
        words = words[:num_words]

        if words:
            reduced_embeddings = compute_embeddings_and_pca(model, words)
            if reduced_embeddings is not None:
                fig = go.Figure(data=[go.Scatter3d(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    z=reduced_embeddings[:, 2],
                    text=words,
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=np.arange(len(words)),
                        colorscale='Viridis',
                        opacity=0.8,
                    ),
                )])

                fig.update_layout(
                    title='Interactive 3D Word Embeddings',
                    scene=dict(
                        xaxis_title='PCA Component 1',
                        yaxis_title='PCA Component 2',
                        zaxis_title='PCA Component 3',
                        xaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                        yaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                        zaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                    ),
                    margin=dict(l=0, r=0, b=0, t=50),
                    hovermode='closest',
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font_color="white",
                    scene_camera=dict(
                        eye=dict(x=1.25, y=1.25, z=1.25)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Not enough words to compute PCA.")
        else:
            st.write("No valid words found in the model.")