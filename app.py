import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2  #  Required for face detection
# updated imports 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
#configuration
st.set_page_config(page_title="Emo-RAG Analyst", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model loading part 
@st.cache_resource
def load_vision_model():
    """Loads the Fine-Tuned ResNet18 Model"""
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7) 
    
    try:
        model.load_state_dict(torch.load("resnet_finetuned.pth", map_location=DEVICE))
    except FileNotFoundError:
        st.sidebar.warning("⚠️ 'resnet_finetuned.pth' not found. Using untrained model for demo.")
    
    model.to(DEVICE)
    model.eval()
    return model

def detect_and_crop_face(image):
    """
    detects a face in the image and crops it. 
    good for human face detection 
    Fixes the issue where background noise confuses the model.
    """
    # Convert PIL Image to Numpy Array (RGB)
    img_array = np.array(image)
    # Convert RGB to BGR (OpenCV standard)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Load Haar Cascade (Built-in to OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Find the largest face (area = w * h)
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        
        # Add a little padding around the face for context
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w += padding * 2
        h += padding * 2
        
        # Crop using PIL logic
        return image.crop((x, y, x + w, y + h)), True
    
    return image, False

def predict_emotion(model, image):
    """Runs inference on a single image"""
    # NEW: Try to crop the face first
    cropped_face, face_found = detect_and_crop_face(image)
    
    if face_found:
        st.sidebar.success("✅ Face Detected & Cropped!")
        st.sidebar.image(cropped_face, caption="Model Input", width=150)
    else:
        st.sidebar.warning("⚠️ No face detected. Using full image.")
    # Same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(cropped_face).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        score, predicted = torch.max(outputs, 1)
        
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[predicted.item()], probabilities[0].cpu().numpy()

#rag pipline

class SimpleRAGChain:
    """Manual RAG implementation with improved prompting."""
    def __init__(self, vector_db, llm):
        self.vector_db = vector_db
        self.llm = llm
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    def invoke(self, inputs):
        query = inputs["query"]
        source_documents = self.retriever.invoke(query)
        unique_contents = list(set([doc.page_content for doc in source_documents]))
        context_text = "\n\n".join(unique_contents)
        
        # prompt for natural summarisation 
        prompt = f"""Use the customer reviews below to answer the question. 
        Do not repeat yourself. Answer in a natural, helpful tone.
        
        Customer Reviews:
        {context_text}
        
        Question: {query}
        
        Answer:"""
        
        result = self.llm.invoke(prompt)
        return {"result": result, "source_documents": source_documents}

@st.cache_resource
def setup_rag_system():
    """Initializes LLM with Repetition Penalties to fix 'rigged' feeling."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # parameters for generation
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=150,
        do_sample=True,          # Adds creativity
        temperature=0.7,         # Makes it less robotic
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.5   #no loop 
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return embeddings, llm

def generate_synthetic_data(emotion):
    """Generates fake reviews based on the detected emotion"""
    templates = {
        'Angry': ["I am furious with the service!", "Worst experience ever.", "I want a refund now."],
        'Happy': ["Absolutely amazing experience!", "The staff was so helpful.", "I am smiling ear to ear."],
        'Sad': ["This really let me down.", "I am disappointed and heartbroken.", "Regrettable purchase."],
        'Surprise': ["Wow! I did not expect this.", "Incredible quality, I am shocked.", "A pleasant surprise!"],
        'Neutral': ["It was okay.", "Nothing special to report.", "Average experience."],
        'Fear': ["I was scared to come here.", "This place gives me anxiety.", "I feel unsafe."],
        'Disgust': ["That was gross.", "Hygiene is terrible.", "I felt sick after this."]
    }
    
    texts = templates.get(emotion, templates['Neutral']) * 3
    docs = [Document(page_content=t, metadata={"emotion": emotion}) for t in texts]
    return docs

# web based ui using streamlit

st.title("🧠 AI Emotion-RAG Analyst")
st.markdown("Upload a face image -> AI detects emotion -> AI generates customer reviews -> You chat with the data.")

st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload a Face Image", type=['jpg', 'png', 'jpeg'])

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    # A. Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(image, caption='Uploaded Image', use_container_width=True)
    
    # B. Run Vision Model (With Face Crop)
    vision_model = load_vision_model()
    emotion, probs = predict_emotion(vision_model, image)
    
    st.sidebar.markdown(f"### Detected Emotion: **{emotion}**")
    st.sidebar.progress(float(probs[np.argmax(probs)]))
    
    # C. Build RAG Knowledge Base
    embeddings, llm = setup_rag_system()
    docs = generate_synthetic_data(emotion)
    vector_db = FAISS.from_documents(docs, embeddings)
    qa_chain = SimpleRAGChain(vector_db, llm)
    
    st.success(f"✅ Knowledge Base Created! Generated synthetic reviews for '{emotion}'.")

    # D. Chat Interface
    st.divider()
    st.subheader("2. Chat with the Analysis")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the customer's reaction..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload an image to start the analysis.")