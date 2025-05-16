#pip install flask numpy pillow tensorflow
import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
# from keras.models import load_model
import uuid

app = Flask(__name__)
model = load_model('PneumoniaPrediction1.keras')  

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    # img = Image.open(image_path).convert('RGB')

    # # Crop center square
    # w, h = img.size
    # min_dim = min(w, h)
    # img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
    #                 (w + min_dim) // 2, (h + min_dim) // 2))

    # img = img.resize((224, 224))
    # img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    # return img_array

    image = cv2.imread(image_path)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (224, 224))
    image1 = img_to_array(image1)
    image1 = preprocess_input(image1)
    image1 = np.expand_dims(image1, axis=0)
    return image1

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    result_file= None
    label=  None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            image = cv2.imread(filepath)

            (normal, pneumonia) = model.predict(img_array)[0]
        label = "normal" if normal > pneumonia else "pneumonia"
        color = (0, 255, 0) if label == "normal" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(normal, pneumonia) * 99.8)
        cv2.putText(image, label, (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        print("RESULT :" +label)

        unique_filename = f"{uuid.uuid4().hex}_prediction.jpg"
        result_file=os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        # file.save(result_file)
        cv2.imwrite(result_file, image)

    return render_template('index.html', prediction=label, image_path=result_file)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
