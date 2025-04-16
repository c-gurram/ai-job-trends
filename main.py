import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
import tempfile
import os

nltk.download("punkt")

# Set up Streamlit app
st.set_page_config(page_title="Job Helper ", layout="wide")
st.title("AI-Driven Job Market Analysis and Forecasting")

# Initialize components
llm = Ollama(model="llama3.2")  # Use a locally installed LLaMA model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None
document_store = []

tabs = st.tabs(["Upload Dataset", "Data Overview", "Insights", "Interactive Q&A", "Word Cloud", "Highlights"])

# Process uploaded dataset
def process_dataset(uploaded_file):
    global vectorstore
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    df = pd.read_excel(temp_file_path)
    text_data = df.to_string(index=False)
    document_store.append(text_data)
    
    if vectorstore is None:
        vectorstore = FAISS.from_texts([text_data], embedding_model)
    else:
        vectorstore.add_texts([text_data])
    
    os.remove(temp_file_path)
    return df

with tabs[0]:
    st.header("Upload Job Dataset")
    uploaded_file = st.file_uploader("Upload an Excel file with job details", type=["xlsx"])
    
    if uploaded_file:
        df = process_dataset(uploaded_file)
        st.success("Dataset processed successfully!")

with tabs[1]:
    st.header("Data Overview")
    if document_store:
        st.dataframe(df.head(10))
    else:
        st.info("Please upload a dataset to display.")

with tabs[2]:
    st.header("Summarization")
    
    if document_store:
        prompt = f"Summarize the key insights from the job dataset:\n\n{document_store[0][:2000]}..."
        summary = llm(prompt)
        st.write("### Key Insights:")
        st.write(summary)
    else:
        st.info("Upload a dataset to summarize.")

with tabs[3]:
    st.header("Interactive Q&A")
    
    if not vectorstore:
        st.info("Upload a dataset to enable Q&A.")
    else:
        question = st.text_input("Ask a question about the job market data:")
        if question:
            prompt = f"Answer based on the dataset:\n\n{question}"
            answer = llm(prompt)
            st.write("### Answer:")
            st.write(answer)

with tabs[4]:
    st.header("Word Cloud")
    if document_store:
        wordcloud = WordCloud(background_color="white", max_words=200).generate(document_store[0])
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.info("Upload a dataset to generate a word cloud.")

with tabs[5]:
    st.header("Highlights")
    if document_store:
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        entities = ner_model(document_store[0][:2000])
        highlights = [f"{e['word']} ({e['entity_group']})" for e in entities]
        st.write("### Key Highlights:")
        st.write(", ".join(set(highlights)))
    else:
        st.info("Upload a dataset to extract highlights.")
