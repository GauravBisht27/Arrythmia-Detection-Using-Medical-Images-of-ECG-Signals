from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        file = request.files.get('file')

        if file:
            img_path = os.path.join('uploads', file.filename)
            file.save(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0) / 255.0
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            
            # Define the classes
            classes = ["ECG Images of Myocardial Infarction Patients",
                       "ECG Images of Patient that have History of MI",
                       "ECG Images of Patient that have abnormal heartbeat",
                       "Normal Person ECG Images"]
            result = classes[predicted_class]
            
            # Define suggestions based on the result
            suggestions = {
                'ECG Images of Myocardial Infarction Patients': 'Immediate medical attention is required. Consult a cardiologist.',
                'ECG Images of Patient that have History of MI': 'Regular check-ups and a healthy lifestyle are essential. Follow prescribed medication.',
                'ECG Images of Patient that have abnormal heartbeat': 'Consult a cardiologist for further evaluation. Lifestyle changes might be needed.',
                'Normal Person ECG Images': 'Maintain a healthy lifestyle to keep your heart healthy.'
            }
            suggestion = suggestions[result]

            return render_template("result.html", name=name, age=age, gender=gender, result=result, suggestions=suggestion)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
