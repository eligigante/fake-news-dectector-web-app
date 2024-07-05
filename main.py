import pandas as pd
import streamlit as st

column1, column2 = st.columns([1, 9])\

with column1:
    st.write("")
    st.image('images/logo_news.png', width = 65)

with column2:
    st.title("Fake News Detector v1.0")

st.text("A simple web application, created using the Streamlit library, that identifies \nwhether a news article is likely to be true or false.")

text = st.text_area("Article: ", placeholder="Input text here")
st.markdown('<p style="font-style: italic; color: gray;">Note: Please note that it may sometimes produce inaccurate results.</p>', unsafe_allow_html=True)

# code for bert or whatever
# if text:


with st.expander("About the creators", icon='ℹ'):
    st.markdown("""
    This web application was created by a team to achieve a proof of concept, and to gain experience in using the Streamlit library, aiming to provide a simple and intuitive tool 
    for identifying fake news. We hope you find it useful!

    Feedback or suggestions are welcome.
    """)
