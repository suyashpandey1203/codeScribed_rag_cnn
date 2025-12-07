# How to run Guide:

## Prerequisites:
* Python: 3.8 or higher

* OS: Windows, Mac, or Linux

* Hardware: CPU is supported (and default). A GPU (NVIDIA CUDA) is recommended for faster training but not required.

## Setup & Installation
Windows:
    
    python -m venv venv
    .\venv\Scripts\activate

linux or mac:
    
    python3 -m venv venv
    source venv/bin/activate
    
## Installation

run this command 
```
 pip install -r requirements.txt
```
## Download the dataset
so will be keeping the dataset in this file so no need to download additionally

# Model Training
Note:
    So this step is not necessary as i will be uploading the model which is pretrained , but in case if curious , then these are the following step's :


* Run the fine-tuning script. This will download the pre-trained ResNet18 and adapt it to the 7 emotion classes.
```
python finetune_train.py
```
* Output: Creates resnet_finetuned.pth (The trained weights). (Stage 1 deliverable )

## Evaluation of the model

* Run this command to get other deliverable of stage 1 that is **confusion_matrix.png** ,**test_predictions.csv (Used by Stage 2 for testing)** 


Also rag can be tested using ```rag.py``` on the terminal

## Working Web app
This is the main product. It integrates the Vision Model and the RAG Chatbot into a single UI.


Run the app using command :
``` streamlit run app.py ```





# Troubleshooting
* "No module named..." errors
Make sure your virtual environment is active ((venv) should appear in your terminal).

Run pip install -r requirements.txt again.

* Model Download Slowness
The first time you run app.py, it downloads the ```flan-t5-large``` model (~1GB). This happens only once. Please be patient.
