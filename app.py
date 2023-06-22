import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the model
model = tf.keras.models.load_model('toxicity.h5')

# Load the data and define X and y
df = pd.read_csv(r"C:\Shivam\Data Science Projects\jigsaw-toxic-comment-classification-challenge\train.csv.zip")
X = df['comment_text']
y = df[df.columns[2:]].values

# Create a TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(max_tokens=200000,
                              output_sequence_length=1800,
                              output_mode='int')
vectorizer.adapt(X.values)

# Create a function to make predictions
def predict(text):
    vectorize_comment = vectorizer([text])
    results = model.predict(vectorize_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '- {}: {}\n'.format(col, results[0][idx]>0.5)
        
    return text

# Create the Streamlit interface
st.title('Toxic Comment Classification')
text = st.text_input('Enter a comment:')
if st.button('Predict'):
    prediction = predict(text)
    st.write(f'Prediction:')
    st.markdown(prediction)
