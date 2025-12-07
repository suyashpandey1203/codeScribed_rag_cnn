"""
-----------------------------------------------------------------------------------------
NOTE 

This file is provided to explicitly satisfy the **Stage 2** requirement for a 
standalone, console-based RAG pipeline using LangChain.

**PRIMARY SUBMISSION:**
The full end-to-end application (Vision + RAG + UI) is located in **`app.py`**.
We recommend running `streamlit run app.py` to see the complete integrated project.

**Why is this file here?**
1. Requirement Fulfillment: It demonstrates the LangChain/RAG logic in isolation as requested in Stage 2.
2. Modular Testing: It allows for testing the NLP pipeline via terminal without loading 
   the heavy Vision models or the Web UI found in `app.py`.
-----------------------------------------------------------------------------------------
"""

import pandas as pd
import random
import os
import torch

# --- UPDATED STABLE IMPORTS ---
# 1. Document Schema
from langchain_core.documents import Document

# 2. Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Vector Store
from langchain_community.vectorstores import FAISS

# 4. LLM
from langchain_community.llms import HuggingFacePipeline

# Transformers for Local LLM & Sentiment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# --- PART 1: DATA GENERATION (Review Synthesis) ---

TEMPLATES = {
    'Angry': [
        "I am absolutely furious! The service was terrible.",
        "Worst experience ever. I will never come back here.",
        "Completely unacceptable behavior from the staff. I'm so mad!",
        "This is a scam. I want my money back immediately."
    ],
    'Happy': [
        "What a wonderful day! The staff was so helpful.",
        "I absolutely loved it! 10/10 would recommend.",
        "Great atmosphere and amazing quality. I'm smiling ear to ear!",
        "Best experience of my life. Thank you so much!"
    ],
    'Sad': [
        "I'm feeling really down about the quality of this product.",
        "It was a disappointing experience that left me heartbroken.",
        "I expected better. This just made my day worse.",
        "Regrettable purchase. I feel let down."
    ],
    'Surprise': [
        "Wow! I did not expect that at all.",
        "Incredible! I was completely shocked by how good this is.",
        "Unexpectedly amazing features. My jaw dropped.",
        "I was surprised by the speed of delivery. Whoa!"
    ],
    'Neutral': [
        "It was okay. Nothing special.",
        "Average experience. Meets expectations but doesn't exceed them.",
        "Standard service. No complaints, no praises.",
        "It is what it is. Fine for the price."
    ],
    'Fear': [
        "I was scared to use this product, it looks dangerous.",
        "The atmosphere was terrifying. I wanted to leave.",
        "Anxiety-inducing experience. Not safe at all.",
        "I'm afraid this won't work as promised."
    ],
    'Disgust': [
        "That was revolting. I felt sick.",
        "Absolutely gross. Hygiene standards are non-existent.",
        "Yuck! I cannot believe they served this.",
        "Repulsive quality. Stay away."
    ]
}

def generate_reviews(csv_path='test_predictions.csv'):
    """Reads predicted emotions and generates synthetic text reviews."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Generating dummy data for demonstration.")
        # Create dummy data if Stage 1 wasn't run
        data = pd.DataFrame({
            'Predicted_Label': ['Happy', 'Angry', 'Neutral', 'Surprise', 'Sad'] * 10
        })
    else:
        data = pd.read_csv(csv_path)

    reviews = []
    print(f"Generating synthetic reviews for {len(data)} predictions...")
    
    for idx, row in data.iterrows():
        emotion = row['Predicted_Label']
        # Fallback for unknown emotions
        if emotion not in TEMPLATES: emotion = 'Neutral'
        
        # Pick a random template and add some noise
        text = random.choice(TEMPLATES[emotion])
        
        # Create a LangChain Document
        doc = Document(page_content=text, metadata={"source": idx, "emotion": emotion})
        reviews.append(doc)
        
    return reviews

# --- PART 2: RAG PIPELINE SETUP (Manual Logic) ---

class SimpleRAGChain:
    """
    A simple class to mimic RetrievalQA without relying on langchain.chains.
    This fixes the 'No module named langchain.chains' error by doing it manually.
    """
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    def invoke(self, inputs):
        query = inputs["query"]
        
        # 1. Retrieve Docs
        source_documents = self.retriever.invoke(query)
        
        # 2. Prepare Context
        context_text = "\n\n".join([doc.page_content for doc in source_documents])
        
        # 3. Create Prompt
        prompt = f"""Answer the question based on the context below. Keep it concise.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Answer:"""
        
        # 4. Generate Answer
        result = self.llm.invoke(prompt)
        
        return {
            "result": result,
            "source_documents": source_documents
        }

def setup_rag_pipeline(documents):
    print("Loading Embedding Model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building Vector Database (FAISS)...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    print("Loading Local LLM (google/flan-t5-small)...")
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=200
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    # Use our robust manual chain instead of the fragile RetrievalQA
    rag_chain = SimpleRAGChain(vector_db, local_llm)
    
    return rag_chain

# --- PART 3: SENTIMENT ANALYSIS ---

def analyze_sentiment(text):
    """Uses a pre-trained BERT pipeline for sentiment analysis."""
    sent_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sent_pipeline(text)[0]
    return result['label'], result['score']

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Generate Data
    docs = generate_reviews('test_predictions.csv')
    print(f"Generated {len(docs)} review documents.")
    
    # 2. Build Pipeline
    rag_chain = setup_rag_pipeline(docs)
    
    print("\n" + "="*50)
    print("AI RAG ANALYST READY")
    print("="*50)
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Enter your query (e.g., 'What are customers angry about?'): ")
        if query.lower() == 'exit':
            break
            
        # 3. Run RAG Query
        print("\nSearching database and generating answer...")
        response = rag_chain.invoke({"query": query})
        
        print(f"\nAI Summary: {response['result']}")
        
        # 4. Show Source Documents & Sentiment
        print("\n--- Retrieved Evidence & Sentiment Analysis ---")
        for i, doc in enumerate(response['source_documents']):
            content = doc.page_content
            emotion = doc.metadata['emotion']
            
            # Run Sentiment Analysis
            sentiment_label, sentiment_score = analyze_sentiment(content)
            
            print(f"[{i+1}] Review: \"{content}\"")
            print(f"    Original Emotion (Vision): {emotion}")
            print(f"    Text Sentiment (NLP): {sentiment_label} ({sentiment_score:.2f})")
            print("-" * 30)