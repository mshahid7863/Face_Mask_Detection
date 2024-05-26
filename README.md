<img align="center" alt="coding" width="4800" height="350" src="https://github.com/mshahid7863/Face_Mask_Detection/blob/main/GUI.png" height="30" width="40">

# Face_Mask_Detection
#### Dataset : https://www.kaggle.com/datasets/shahidd/face-mask-ds
#### Smaple Datase :https://www.kaggle.com/datasets/shahidd/samples


### Objective:
The project aims to develop a reliable and efficient face mask detection system to help curb the spread of COVID-19 by ensuring compliance with mask-wearing guidelines. The system uses deep learning models for accurate detection and is deployed both as a web application using Gradio and as a real-time video stream.

## Models Used:

1.VGG16:

A pre-trained convolutional neural network (CNN) known for its deep architecture and effectiveness in image classification tasks.
Fine-tuned on a custom dataset of face images with and without masks. 

2.ResNet50:

Another pre-trained CNN that uses residual learning to ease the training of very deep networks.
Fine-tuned similarly to VGG16 for enhanced mask detection performance.
Custom CNN Model:

3.A CNN designed specifically for this task, optimized for speed and accuracy in detecting masks in various conditions.

### Deployment:

1.Gradio Web Interface:

An easy-to-use web interface allowing users to upload images for mask detection.
Users receive instant feedback on whether the person in the image is wearing a mask or not.

2.Live Video Deployment:

Utilizes OpenCV to capture live video from a webcam.
Real-time detection of faces and mask status, with bounding boxes and labels displayed on the video feed.
Achieves an accuracy rate of 95%, making it highly reliable for practical use.

Accuracy:

The models were trained and tested on a diverse dataset to ensure robustness and high accuracy.

The combined system achieves 95% accuracy in detecting whether individuals are wearing masks, demonstrating its effectiveness in real-world scenarios.

Key Features:

* High Accuracy: Achieves 95% accuracy using advanced deep learning models.

* Real-Time Detection: Capable of processing live video streams to detect masks in real-time.

* User-Friendly Interface: Deployed using Gradio for easy interaction and accessibility.

* Versatile: Combines the strengths of VGG16, ResNet50, and a custom CNN to ensure robust performance across different environments.

Conclusion:

This project provides a comprehensive solution for face mask detection using state-of-the-art deep learning techniques. By leveraging powerful models and deploying them through both web interfaces and live video streams, it offers a practical tool to assist in the fight against COVID-19, ensuring public health and safety with high accuracy and reliability.


## Output
<img align="left" alt="coding" width="600"  src="https://github.com/mshahid7863/Face_Mask_Detection/blob/main/Withmask.png" >  

<img align="right" alt="coding" width="600"  src="https://github.com/mshahid7863/Face_Mask_Detection/blob/main/withoutmask.png" >
