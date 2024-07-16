import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from textblob import TextBlob
import streamlit as st
from googletrans import Translator
from langdetect import detect

class CustomBertModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(CustomBertModel, self).__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(bert_model.config.hidden_dropout_prob)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask, polarity = inputs
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = bert_outputs.logits
        concatenated = tf.concat([logits, tf.expand_dims(polarity, -1)], axis=-1)
        output = self.dense(self.dropout(concatenated))
        return output

# Load the original BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('model/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the custom BERT model
base_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
custom_model = CustomBertModel(base_model)
custom_model.load_weights('Bert custom model')

# Layout
column1, column2 = st.columns([1, 9])
with column1:
    st.write("")
    st.image('images/logo_news.png', width=65)
with column2:
    st.title("Fake News Detector v1.0")

st.text("A simple web application, created using the Streamlit library, that identifies \nwhether a news article is likely to be true or false.")

text = st.text_area("Article: ", placeholder="Input text here")
st.markdown('<p style="font-style: italic; color: gray;">Note: Please note that it may sometimes produce inaccurate results.</p>', unsafe_allow_html=True)

# Buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    analyze_sentiment = st.button('Analyze Sentiment')
with col2:
    detect_news = st.button('Detect Fake News')
with col3:
    analyze_both = st.button('Analyze Both')

def preprocess_text(text):
    return text.lower().strip()

def check_fake_news_bert(input_text):
    preprocessed_text = preprocess_text(input_text)
    inputs = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=42, return_tensors='tf')

    token_tensors = inputs['input_ids']
    segment_tensors = inputs['token_type_ids']
    mask_tensors = inputs['attention_mask']

    predictions = model.predict([token_tensors, segment_tensors, mask_tensors])
    logits = predictions.logits[0]
    probabilities = tf.nn.softmax(logits)
    predicted_label = tf.argmax(probabilities).numpy()

    if predicted_label == 0:
        return 'fake', probabilities
    else:
        return 'real', probabilities

def sentiment_analysis(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return polarity

def predict_news(input_text):
    preprocessed_text = preprocess_text(input_text)
    sentiment = sentiment_analysis(preprocessed_text)
    polarity = sentiment
    inputs = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=42, return_tensors='tf')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    polarity_tensor = tf.convert_to_tensor([polarity], dtype=tf.float32)
    predictions = custom_model([input_ids, token_type_ids, attention_mask, polarity_tensor])
    probabilities = tf.nn.softmax(predictions[0])
    predicted_label = tf.argmax(probabilities).numpy()
    return predicted_label, probabilities.numpy()

# Checks language and translates to english
translator = Translator()
if text:
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translated_text = translator.translate(text, dest='en').text
            st.write('Translated Text:', translated_text)
        else:
            translated_text = text
    except Exception as e:
        st.error(f"Error in translation: {e}")

    if analyze_sentiment or analyze_both:
        # Sentiment analysis
        polarity = sentiment_analysis(translated_text)
        st.write('Polarity:', polarity)
        subjectivity = round(TextBlob(translated_text).sentiment.subjectivity, 2)
        st.write('Subjectivity:', subjectivity)

        if polarity >= 0.1:
            st.markdown('<div style="text-align:center;"><span style="font-size:100px;">😊</span><p style="font-size:24px;">Happy!</p></div>', unsafe_allow_html=True)
        elif polarity <= -0.1:
            st.markdown('<div style="text-align:center;"><span style="font-size:48px;">😞</span><p style="font-size:24px;">Sad!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;"><span style="font-size:48px;">😐</span><p style="font-size:24px;">Neutral!</p></div>', unsafe_allow_html=True)

    if detect_news:
        # Fake news detection
        result, probabilities = check_fake_news_bert(translated_text)
        if result == 'fake':
            st.markdown('<div style="text-align:center;"><span style="font-size:100px;">❌</span><p style="font-size:24px;">Fake News!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;"><span style="font-size:100px;">✅</span><p style="font-size:24px;">Real News!</p></div>', unsafe_allow_html=True)
        st.write(f"Probability of being fake: {probabilities[0].numpy():.2%}")
        st.write(f"Probability of being real: {probabilities[1].numpy():.2%}")

    if analyze_both:
        # Fake news detection with custom model
        predicted_label, probabilities = predict_news(translated_text)
        if predicted_label == 0:
            st.markdown('<div style="text-align:center;"><span style="font-size:100px;">❌</span><p style="font-size:24px;">Fake News!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;"><span style="font-size:100px;">✅</span><p style="font-size:24px;">Real News!</p></div>', unsafe_allow_html=True)
        st.write(f"Probability of being fake: {probabilities[0]:.2%}")
        st.write(f"Probability of being real: {probabilities[1]:.2%}")

with st.expander("About the creators", icon='ℹ'):
    st.markdown("""
    This web application was created by MAI-Team to meet the requirements of their Final Project in Data Mining
    """)
