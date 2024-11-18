import streamlit as st
import joblib
import nltk
from preprocessing import TextPreprocessor  # Assuming you have a custom preprocessing class


# Ensure the stopwords are downloaded
nltk.download('stopwords')

# Ensure the stopwords are downloaded
nltk.download('stopwords')

# Load the trained machine learning pipeline
pipeline = joblib.load('models/sentiments_pipeline.pkl')

# Streamlit app configuration
st.set_page_config(page_title="Mental Well-Being Screening", page_icon="🧠", layout="wide")

# Title and description
st.title("Mental Well-Being Screening Tool 🧠")
st.write("""
This application analyzes social media posts from Reddit to predict whether a user is likely experiencing depression or not. It leverages 
         a pre-trained machine learning model for advanced text analysis
""")

# Input section
st.header("Enter Reddit Text for Analysis")
user_input = st.text_area(
    "Paste your text here (e.g., Reddit title entry or Reddit body entry):",
    placeholder="Type or paste text here...",
    height=150
)

# Button to make predictions
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter some text for analysis.")
    else:
        # Use the loaded pipeline to predict
        prediction = pipeline.predict([user_input])[0]
        probability = pipeline.predict_proba([user_input])[0]

        # Interpret the prediction
        result = "Depressed" if prediction == 1 else "Not Depressed"
        confidence = probability[int(prediction)] * 100  # Ensure prediction is used as an integer index

        # Display the results
        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")

# Sidebar
st.sidebar.header("About the App")
st.sidebar.write("""
This tool was created as part of a machine learning project to predict user mental wellness as either depressed or not depressed
from text from Reddit posts. 
It utilizes **TF-IDF** for text vectorization and a logistic regression model for classification.
""")
st.sidebar.write("The dataset used for training includes labeled social media posts.")

# Footer
st.write("---")
st.write("Developed by Gicheha_W. Powered by Streamlit and Scikit-learn.")
