from flask import *
import cv2
from classifiers import *
import base64
import numpy as np
import pickle
from cnn import *

app = Flask(__name__)
app.secret_key = 'S3CR3TK3Y'

cnn_model = CNN()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # If there is a GET request, display the index page
        return render_template('index.html')
    elif request.method == 'POST':
        # if there is a POST request, gather the inputs and send the page back
        # after classifying the image from the backend

        # TODO: Get the "drawn image" and preprocess here (shape=(784, 1))

        pred = None
        draw = request.form['url']
        model_chosen = request.form['model_chosen']
        print(model_chosen)
        # remove the url from the image
        draw = draw[21:]

        # DECODE - Get the image data
        draw_decode = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decode), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # resize and reshape the input (784, 1) -> values between 0 to 1
        resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

        vect = np.asarray(resized, dtype='uint8')
        vect = vect.reshape(28, 28).astype('float32')
        vect = vect.reshape((784, 1))
        vect = vect / 255.0

        nn_data = vect.reshape((-1, 28, 28, 1))

        bayes = classify_naive_bayes(vect)
        svm = classify_svm_one_vs_one(vect)

        # TODO: Classify the image
        pred = cnn_model.predict(nn_data)

        if model_chosen is not None and model_chosen == 'Naive Bayes':
            # update the predicition if the model chosen is Naive Bayes
            pred = bayes
        elif model_chosen is not None and model_chosen == 'SVM (One vs One)':
            pred = svm

        # label = bayes
        label = pred
        return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run()
