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

model = ResNet50(weights="imagenet",input_shape=(224,224,3))
model_new = Model(model.input,model.layers[-2].output)

# with open("saved/encoded_train_features.pkl", "rb") as f:
#     encoding_train = pickle.load(f)
# with open("saved/encoded_test_features.pkl", "rb") as f:
#     encoding_test = pickle.load(f)
    
model_path="model_weights\model_19.h5"
model_f = load_model(model_path)

index_to_word = {}
with open("saved\index_to_word.pkl","rb") as f:
    index_to_word=pickle.load(f)
    
word_to_index = {} 
with open("saved\word_to_index.pkl","rb") as f:
    word_to_index=pickle.load(f)

max_len = 35

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    print(feature_vector.shape)
    return feature_vector

def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        # print("Sequence shape:", sequence.shape)  # Checking shape of sequence 
        ypred = model_f.predict([photo, sequence])
        # print("Prediction shape:", ypred.shape)  # Checking prediction output shape
        ypred = ypred.argmax()
        word = index_to_word.get(ypred, None)
        if word is None:
            # print(f"Warning: predicted index {ypred} is not in index_to_word.")
            break
        in_text += (' ' + word)
        if word == "endseq":
            break
    
    final_caption = ' '.join(in_text.split()[1:-1])
    return final_caption


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join("static/uploaded_images", file.filename)
        file.save(file_path)

        photo = encode_image(file_path)
        photo_2048=photo.reshape((1,2048))
        caption = predict_caption(photo_2048)

        return jsonify({"image_url": file_path, "caption": caption})
    except Exception as e:
        print("Error encountered:", e)
        return jsonify({"error": "An error occurred on the server."}), 500


if __name__ == '__main__':
    app.run(debug=True)