from flask import Flask, request, render_template, jsonify
from keras.models import load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)
app.config['DEBUG'] = True

# Load the encoding
with open("saved\encoded_train_features.pkl", "rb") as f:
    encoding_train = pickle.load(f)
    
# Load word mappings
word_to_index = {} 
with open("saved\index_to_word.pkl","rb") as f:
    index_to_word=pickle.load(f)
    
index_to_word = {}
with open("saved\word_to_index.pkl","rb") as f:
    word_to_index=pickle.load(f)

max_len = 35

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

model = ResNet50(weights="imagenet",input_shape=(224,224,3))
model_new = Model(model.input,model.layers[-2].output)
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    print(feature_vector.shape)
    return feature_vector

# Load model
model_path = "../model_weights/model_19.h5"
model = load_model(model_path)

def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]
        in_text += ' ' + word
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file
    file_path = os.path.join("static/uploaded_images", file.filename)
    file.save(file_path)

    # Process the image and generate caption
    img = Image.open(file_path)
    photo = encode_image(img)
    photo_2048=photo.reshape((1,2048))
    caption = predict_caption(photo_2048)

    return jsonify({"caption": caption})
