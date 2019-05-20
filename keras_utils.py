from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image as keras_image
import os
import numpy as np
from utils import Camera, projectModel

class KerasDataGenerator(Sequence):

    """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

    def __init__(self, label_list, speed_root, label_size, batch_size=32, dim=(224, 224), shuffle=True):

        # loading dataset
        self.image_root = os.path.join(speed_root, 'images', 'train')

        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                                     for label in label_list}
        self.list_IDs = [label['filename'] for label in label_list]
        self.shuffle = shuffle
        self.label_size = label_size
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):

        """ Denotes the number of batches per epoch. """

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        """ Generate one batch of data """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):

        """ Updates indexes after each epoch """

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def drawBlob(self, img, pos, sigma=3):
        # https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/training/train.py#L851
        w = int(sigma*3)
        if pos[0]-w>=0 and pos[0]+w<img.shape[0] and pos[1]-w>=0 and pos[1]+w<img.shape[1]:
            for i in range(int(pos[0])-w, int(pos[0])+w):
                for j in range(int(pos[1])-w, int(pos[1])+w):
                    img[i,j] = np.exp(-(((i - pos[0])**2 + (j - pos[1])**2)/(2*(sigma**2))))

    def __data_generation(self, list_IDs_temp):

        """ Generates data containing batch_size samples """

        # Initialization
        imgs = np.empty((self.batch_size, *self.dim, 1))
        masks = np.zeros((self.batch_size, *self.dim, 8), dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img_path = os.path.join(self.image_root, ID)
            img = keras_image.load_img(img_path, target_size=self.dim, color_mode = "grayscale")
            imgs[i] = keras_image.img_to_array(img)

            q, r = self.labels[ID]['q'], self.labels[ID]['r']
            xa, ya, visibles = projectModel(q, r)
            for j, (x, y, visible) in enumerate(zip(xa, ya, visibles)):
                x /= Camera.nu
                y /= Camera.nv
                if visible and x >= 0.0 and y >= 0.0 and x <= 1.0 and y <= 1.0:
                    x_s, y_s = int(x * self.dim[1]), int(y * self.dim[0])
                    self.drawBlob(masks[i][...,j], (x_s, y_s), self.label_size)

        return imgs, masks
