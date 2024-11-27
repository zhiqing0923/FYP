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
        self.scale = 0 
        self.x_offset=0
        self.y_offset=0

    def process_model(self,normalized_image):
        image = tf.convert_to_tensor(normalized_image)

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        model = load_model('vgg19_model_11.h5')

        predicted_keypoints = model.predict(image)
        predicted_keypoints = tf.reshape(predicted_keypoints, (image.shape[0], -1, 2))

        # Convert TensorFlow tensor to NumPy array for modification
        predicted_keypoints = predicted_keypoints.numpy()

        # Denormalize the keypoints coordinates to range [0, original_width] and [0, original_height]
        predicted_keypoints = predicted_keypoints * np.array([self.original_width, self.original_height])

        # Adjust keypoints for the cropping offset: shift x-coordinates by x_offset
        # After cropping, the new x-coordinates should reflect the cropped image
        predicted_keypoints[:, 0] = predicted_keypoints[:, 0] - self.x_offset  # Correct x-coordinate by subtracting x_offset

        # Now rescale keypoints to match the resized image dimensions (1935, 2400)
        scale_x = 1935 / (self.original_width - 2 * self.x_offset)  # The width after cropping
        scale_y = 2400 / self.original_height  # The full height remains the same


        predicted_keypoints[:, 0] = predicted_keypoints[:, 0] * scale_x
        predicted_keypoints[:, 1] = predicted_keypoints[:, 1] * scale_y
        
        return predicted_keypoints 

    def process_image(self,image):
        self.original_width = image.shape[1]
        self.original_height = image.shape[0]
        

        # Compute the scaling factors for width and height based on the smaller dimension
        scale_width = self.target_width / self.original_width
        scale_height = self.target_height / self.original_height
        self.scale = min(scale_width, scale_height)

        # Compute the new width and height after scaling
        new_width = int(self.original_width * self.scale)
        new_height = int(self.original_height * self.scale)

        # Resize the image using cv2.resize function with interpolation=cv2.INTER_AREA
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Crop the resized image to match the target width and height from the center
        cropped_image = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        self.x_offset = (self.target_width - new_width) // 2
        self.y_offset = (self.target_height - new_height) // 2
        cropped_image[self.y_offset:self.y_offset + new_height, self.x_offset:self.x_offset + new_width] = resized_image
        
        # Normalize the cropped image pixels to range [0, 1]
        normalized_image = cropped_image / 255.0
        
        keypoints=self.process_model(normalized_image)
        keypoints_squeezed = np.squeeze(keypoints)

        keypoints_squeezed = [tuple(map(int, coord)) for coord in keypoints_squeezed]

        
        return keypoints_squeezed
      

    
    
       
    

    




    