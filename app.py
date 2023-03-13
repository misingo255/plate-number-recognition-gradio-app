import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import imutils
import easyocr
import gradio as gr


def classify(image):

    IMAGE_SHAPE = (224, 224)

    buffer = np.array(image)/255.0

    buffer = buffer[np.newaxis, ...]

    classifier = tf.keras.Sequential([hub.KerasLayer("models/", input_shape=IMAGE_SHAPE+(3,))])

    results = classifier.predict(buffer)

    predicted_image_index = np.argmax(results)

    image_labels = []
    with open("./static/labels.txt", "r") as f:
        image_labels = f.read().splitlines()

    class_name = image_labels[predicted_image_index]

    if class_name == "plate":

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction

        edged = cv2.Canny(bfilter, 30, 200) #Edge detection

        #Finding contours and applying mask
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(keypoints)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(gray.shape, np.uint8)

        (x,y) = np.where(mask==255)

        # Set default values for the minimum and maximum indices
        x1, y1, x2, y2 = 0, 0, mask.shape[0]-1, mask.shape[1]-1

        if len(x) > 0 and len(y) > 0:
            x1, y1 = np.min(x), np.min(y)
            x2, y2 = np.max(x), np.max(y)

        cropped_image = gray[x1:x2+1, y1:y2+1]


        #Apply OCR
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        license_numbers = []
        for value in result:
            license_numbers.append(value[1])

        values = " ".join(license_numbers)

        values = str(values)

        values = values.upper()

        message = "The recognized plate license number is: "+values

        return message
    else:
        return "No License Plate Seems to be found on the picture!!"
    

app = gr.Interface(fn=classify, inputs=gr.Image(shape=(224,224)), outputs="text", title="Image Recognition", description="This project showcases the ability to classify images and recognize texts written on them (e.g Vehicle License Plates Numbers) using machine learning")

app.launch()

