from styx_msgs.msg import TrafficLight
import rospy
import os
import yaml

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

IMG_H = 600   # image height
IMG_W = 800  # image width
IMG_C = 3

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_dir_path = None
        self.model = None
        self.graph = None
        
        # load configuration string
        conf_str = rospy.get_param ("/traffic_light_config")
        self.configuration = yaml.load(conf_str)
        self.model_dir_path = './light_classification/model.h5'
   
        rospy.loginfo ("model directory path: {} ".format(self.model_dir_path))
    
        #load the model
        if  not (os.path.exists(self.model_dir_path)):
            rospy.logerr ("model directory path {} does not exist".format (self.model_dir_path))
        else:
            self.model = load_model(self.model_dir_path)
            
            self.model._make_predict_function()
            self.graph = K.tf.get_default_graph()
            rospy.loginfo ("model loaded successfully from {}".format (self.model_dir_path))
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        try:
            with self.graph.as_default ():
                # check if graph
                if self.graph == None:
                    rospy.logerr ("Graph is None")
                    return TrafficLight.UNKNOWN
            
                # check if model
                if self.model == None:
                    rospy.logerr ("Model is None")
                    return TrafficLight.UNKNOWN
                
                #Adjust the size of the image according to the acceptability of the model
                img = np.reshape (image,  (1, IMG_H, IMG_W, IMG_C))
                score_list = self.model.predict (img)
            
                #check score_list
                if (type (score_list) == None or len(score_list) == 0):
                    rospy.loginfo ("Prediction score list empty")
                    return TrafficLight.UNKNOWN

                light_type = np.argmax (score_list)
                if (light_type == 0):
                    return TrafficLight.RED
                elif (light_type == 1):
                    return TrafficLight.GREEN
                else:
                    return TrafficLight.UNKNOWN
                    
        except Exception as e:
            rospy.logerr ("Traffic Classifier raised exception")
            rospy.logerr (e)
            return TrafficLight.UNKNOWN