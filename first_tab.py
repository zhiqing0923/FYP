# Sample coordinates (as if processed by a model)
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class AI_landmarks(): 

    def __init__(self):
        self.original_height= 0
        self.original_width= 0
        self.target_width= 224
        self.target_height= 224
        self.cropped_image_width=0
        self.cropped_image_height=0
        self.min_x=0
        self.min_y=0


    #Function to get the bounding box coordinates 
    def process_model_1(self,normalized_image):

        # Path to the TFLite model file
        tflite_model_path = "bounding_box_6.tflite"

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

        # Allocate tensors (initialize model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure the input tensor is float32
        normalized_image = normalized_image.astype('float32')

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], normalized_image)

        # Run inference
        interpreter.invoke()
        bounding_box_coordinates = interpreter.get_tensor(output_details[0]['index'])
        

        box_keypoint = bounding_box_coordinates[0] * np.array([self.original_width, self.original_height, self.original_width, self.original_height])
        
        return box_keypoint 
    
    #Function to get 19 keypoints 
    def process_model_2(self,normalized_image):
        

        # Path to the TFLite model file
        tflite_model_path = "vgg19_keypoints.tflite"

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

        # Allocate tensors (initialize model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        

        # Ensure the input tensor is float32
        normalized_image = normalized_image.astype('float32')

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], normalized_image)

        # Run inference
        interpreter.invoke()
        keypoints = interpreter.get_tensor(output_details[0]['index'])
        

        
        
        keypoints = keypoints[0]
        keypoints = keypoints.reshape(-1, 2)

        #denormalize keypoints and rescale it back to original dimension
        keypoints = keypoints * np.array([self.cropped_image_width, self.cropped_image_height])
        keypoints = keypoints + np.array([self.min_x, self.min_y])

        

        return keypoints 

    def process_image(self,image):
        self.original_width = image.shape[1]
        self.original_height = image.shape[0]
        

        # Resize the image using cv2.resize function with interpolation=cv2.INTER_AREA
        resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
       
        if len(resized_image.shape) == 3:
            resized_image = np.expand_dims(resized_image, axis=0)
       
        
        # Normalize the cropped image pixels to range [0, 1]
        normalized_image = resized_image / 255.0
        
        #Get bounding_box coordinates
        bounding_box_coordinates=self.process_model_1(normalized_image)

        min_x, min_y, max_x, max_y= bounding_box_coordinates

        # Ensure indices are integers
        min_y = int(min_y)
        max_y = int(max_y)
        min_x = int(min_x)
        max_x = int(max_x)


        # Crop the image with the bounding_box_coordinates 
        cropped_image = image[min_y:max_y, min_x:max_x]

        #Store data 
        self.min_x=min_x
        self.min_y=min_y
        self.cropped_image_width = cropped_image.shape[1]
        self.cropped_image_height = cropped_image.shape[0]

        # Resize the cropped image
        cropped_resized_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_AREA)

        if len(cropped_resized_image.shape) == 3:
            cropped_resized_image = np.expand_dims(cropped_resized_image, axis=0)

        # Normalize the image to the range [0, 1] if required by the model
        cropped_normalized_image = cropped_resized_image / 255.0
    

        #Get bounding_box coordinates
        keypoints=self.process_model_2(cropped_normalized_image)
        

        keypoints_squeezed = np.squeeze(keypoints)

        keypoints_squeezed = [tuple(map(int, coord)) for coord in keypoints_squeezed]
        
        return keypoints_squeezed
      

    
    
       
    

    




    