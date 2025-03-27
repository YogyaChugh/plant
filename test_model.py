import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/model_multiclass.h5')

# Define the class labels
class_labels =['Apple___Scab',
 'Apple___Black_Rot',
 'Apple___Cedar_Apple_Rust',
 'Apple___Healthy',
 'Cherry___healthy',
 'Cherry___Powdery_Mildew',
 'Corn___Cercospora_Leaf_Spot_Gray_Leaf_Spot',
 'Corn___Healthy',
 'Corn___Common_Rust',
 'Corn___Northern_Leaf_Blight',
 'Grape___Black_Rot',
 'Grape___Esca_Black_Measles',
 'Grape___Healthy',
 'Grape___Leaf_Blight_Isariopsis_Leaf_Spot',
 'Peach___Healthy',
 'Peach___Bacterial_Spot',
 'Pepper_Bell_Bacterial_Spot',
 'Pepper_Bell_Healthy',
 'Potato___Early_Blight',
 'Potato___Healthy',
 'Potato___Late_Blight',
 'Strawberry___Healthy',
 'Strawberry___Leaf_Scorch',
 'Tomato___Bacterial_Spot',
 'Tomato___Healthy',
 'Tomato___Early_Blight',
 'Tomato___Late_Blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_Leaf_Spot',
 'Tomato___Spider_Mites_Two_Spotted_Spider_Mite',
 'Tomato___Target_Spot',
 'Tomato___Mosaic_Virus',
 'Tomato___Yellow_Leaf_Curl_Virus']


def aibro(img_path):
 img = image.load_img(img_path, target_size=(256, 256))  # Resize image
 img_array = image.img_to_array(img) / 255.0  # Convert to array and rescale
 img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

 # Make a prediction
 prediction = model.predict(img_array)
 predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability
 predicted_label = class_labels[predicted_class]
 return f"{predicted_label}"

