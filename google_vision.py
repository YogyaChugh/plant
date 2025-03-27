import tensorflow as tf
from google.cloud import vision
import numpy as np
from PIL import Image
import io
from test_model import aibro
from google.cloud import vision








# Function to call Google Vision API to detect disease
def detect_with_vision_api(image_path):
    # Initialize the Google Vision client
    client = vision.ImageAnnotatorClient()

    # Open the image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Perform label detection on the image
    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    # If the response is successful, parse the results
    if response.error.message:
        print(f"Google Vision API error: {response.error.message}")
        return None

    labels = response.label_annotations

    # Return the first label as the detected disease name
    # You may want to filter or adjust based on your needs
    if labels:
        return labels[0].description
    else:
        return "No disease detected by Vision API"


# Main function that combines both the model prediction and Google Vision API fallback
def predict_disease(image_path):
    try:
        # Try predicting with your model
        predicted_disease=aibro(image_path)
        return predicted_disease

    except Exception as e:
        print(f"Model prediction failed: {e}")
        # If model prediction fails, use Google Vision API as fallback
        predicted_disease = detect_with_vision_api(image_path)
        print(f"Predicted by Google Vision API: {predicted_disease}")
        return predicted_disease


# Example usage
if __name__ == "__main__":

    image_path = "path_to_plant_image.jpg"  # Path to the plant image

    # Load your model


    # Predict disease
    disease = predict_disease('C:/Users/mudit/PycharmProjects/Agro_lens/data/test/Potato___Early_Blight/image (45).JPG')
    print(f"The disease detected is: {disease}")