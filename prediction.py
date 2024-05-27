import cv2
import tensorflow as tf
import os
import numpy as np
test_folder_path ="C:\\Users\\Dipesh Singh\\Downloads\\test\\test"

dir_path1 = test_folder_path+"\\ECG Images of Myocardial Infarction Patients (240x12=2880)"
dir_path2 = test_folder_path+"\\ECG Images of Patient that have History of MI (172x12=2064)"
dir_path3 = test_folder_path+"\\ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
dir_path4 = test_folder_path+"\\Normal Person ECG Images (284x12=3408)"

# model = tf.keras.models.load_model("C:\\Users\\Dipesh Singh\\Desktop\\Arrythmia\\model.h5")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.h5')
print("Current directory:", current_dir)
print("Model path:", model_path)

model = tf.keras.models.load_model(model_path)



# Testing the model
for i,dir_path in enumerate([dir_path1,dir_path2,dir_path3,dir_path4]):
    k=15
    for img_name in os.listdir(dir_path):
        k=k-1
        if k==0:
          break
        print(img_name)
        img_path = os.path.join(dir_path, img_name)
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0) / 255.0
        prediction = model.predict(img)
        # print(prediction)
        predicted_class = np.argmax(prediction)
        print(predicted_class)
        # class_name = list(validation_generator.class_indices.keys())[predicted_class]
        print(f"Image: {img_name},{predicted_class}","Original: ",i)