import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import gradio as gr
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('Vgg16_model.keras')

labels = {'withmask': 0, 'withoutmask': 1 }
index_to_labels = {v: k for k, v in labels.items()}
index_to_labels



def prepare_image(img_pil):
    """Preprocess the PIL image to fit your model's input requirements."""
    # Convert the PIL image to a numpy array with the target size
    img = img_pil.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array /= 255.0  # Rescale pixel values to [0,1], as done during training
    return img_array



# Define the Gradio interface
def predict_face(image):
    # Preprocess the image
    processed_image = prepare_image(image)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    # Get the emotion label with the highest probability
    #predicted_class = np.argmax(prediction, axis=1)
    if prediction[0][0] > 0.5:
        return 'withoutmask'
    else:
        return  'withmask'



interface = gr.Interface(
    fn=predict_face,  # Your prediction function
    inputs=gr.Image(type="pil"),  # Input for uploading an image, directly compatible with PIL images
    outputs="text",  # Output as text displaying the predicted emotion
    title="Face Mask Detection",
    description="Upload an image and see the predicted face."
)

# Launch the Gradio interface
interface.launch()