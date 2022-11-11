import tensorflow as tf
import cv2
import os
import numpy as np

class HandwritingRecognition:
    def __init__(self):
        self.path = os.getcwd();
        self.testingPath = self.path+"/data/testing";
        self.saveModel = self.path+"/data/model";

    def load_model(self):
        self.model = tf.keras.models.load_model(self.saveModel);

    def predict_image(self):
        for el in os.listdir("{}/testing".format(self.testingPath)):
            self.image = tf.keras.preprocessing.image.load_img("{}/testing/{}".format(self.testingPath, el));

            self.X = tf.keras.preprocessing.image.img_to_array(self.image);
            self.X = np.expand_dims(self.X, axis=0);

            self.imageTesting = np.vstack([self.X]);

            self.predict = self.model.predict(self.imageTesting);

            return self.print_result();

    def print_result(self):
        if self.predict == 0:
            return "Imagem Analisada\nResultado: Letra 'i' localizada";
        else:
            return "Imagem Analisada\nResultado: Letra 'i' não localizada";