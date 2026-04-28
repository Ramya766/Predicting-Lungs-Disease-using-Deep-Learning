from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load trained model
model = load_model("model_vgg16.h5")

# Image path
img_path =r"C:\Users\Ramya\OneDrive\Desktop\Lung_Disease_Pred\NORMAL\NORMAL2-IMG3.jpeg";
# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
x = image.img_to_array(img)

# Add batch dimension
x = np.expand_dims(x, axis=0)

# Preprocess image
x = preprocess_input(x)

# Predict
prediction = model.predict(x)

normal_prob = prediction[0][0]
pneumonia_prob = prediction[0][1]

print("NORMAL Probability:", normal_prob)
print("PNEUMONIA Probability:", pneumonia_prob)

if normal_prob > pneumonia_prob:
    print("NORMAL")
else:
    print("PNEUMONIA")