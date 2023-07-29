from flask import Flask, render_template, url_for, request
import numpy as np
import keras
from keras.models import load_model
import tensorflow_hub as hub


# load the model
model = load_model("ann_undersample_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

app = Flask(__name__)

@app.route('/')
def home():
	bg_image = url_for('static', filename='images/eduardo-soares-utWyPB8_FU8-unsplash.jpg')
	return render_template('home.html', bg_image=bg_image)

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		bg_image = url_for('static', filename='images/eduardo-soares-utWyPB8_FU8-unsplash.jpg')
		me = request.form['message']
		message = [float(x) for x in me.split(",")]
		vect = np.array(message).reshape(1, -1)
		my_prediction = model.predict(vect)
		prediction_label = (my_prediction> 0.5).astype("int32").flatten()
		
	return render_template('result.html',prediction = prediction_label, bg_image=bg_image)


if __name__ == '__main__':
	app.run(debug=True)