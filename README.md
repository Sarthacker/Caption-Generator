# 📜 Image Caption Generator

## Overview
This project implements a deep learning neural network that takes images as input and generates descriptive captions. The model combines the power of **Computer Vision** and **Natural Language Processing (NLP)** to interpret visual data and generate contextual language outputs. By leveraging pre-trained models and fine-tuning them with a custom dataset, the project achieves an efficient and scalable solution for automatic image captioning.

## 🚀 Key Features
- **Image Feature Extraction using ResNet50**: The model employs the ResNet50 architecture, a well-known pre-trained convolutional neural network (CNN), for image feature extraction. ResNet50 is used to process the input images, extracting rich visual features that are essential for understanding the content of the image.

- **Caption Generation with LSTM**: The extracted image features are then passed through a Long Short-Term Memory (LSTM) network to generate a descriptive caption. LSTM is utilized to handle the sequential nature of natural language, ensuring that the generated captions are both grammatically correct and contextually appropriate.

- **TensorFlow and Keras Libraries**: The project is built using TensorFlow and Keras, two powerful deep learning frameworks that provide the flexibility and scalability required to develop and train neural networks efficiently.

## 🏗️ Model Architecture

### ResNet50 for Feature Extraction:
- The pre-trained ResNet50 network is used to extract image features. The final fully connected layers are removed, and the output is a 4096-dimensional vector representing the image.

### LSTM for Caption Generation:
- The image features are combined with the text data (captions) and passed into an LSTM network to generate captions word by word. The model is trained to predict the next word in the caption sequence given the image features and previous words.

### Word Embeddings and Tokenization:
- The captions are tokenized and transformed into word embeddings using Word2Vec and GloVe embeddings, helping the model learn semantic relationships between words.

## 📂 Dataset
The model was trained using a dataset of images and corresponding captions. Examples of such datasets include:

- **Flickr8k**: This dataset contains thousands of images, each annotated with multiple descriptive captions. They are commonly used in image captioning tasks.

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://https://github.com/Sarthacker/Caption-Generator.git
cd Caption-Generator
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Move to Flask Directory
```sh
cd Flask
```

### 5️⃣ Run the Development Server
```sh
python app.py
```
