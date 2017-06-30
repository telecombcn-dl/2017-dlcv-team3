from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from skimage import color, exposure, transform, io
import os
import numpy as np
import glob
from keras import backend as K
from inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
K.set_image_data_format('channels_first')
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
K.set_image_data_format('channels_first')

batch_size = 16
num_classes = 13
epochs = 50
data_augmentation = False
img_width = 256
img_height = 256

top_model_weights_path = 'imagenet_fc_model.h5'
path1 = '/home/dlcv/DeployedProjects/Jam/dlcv/train/images'
nb_train_samples = 624
nb_validation_samples = 270
nb_total_samples = 894
epochs = 200
batch_size = 16


def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (img_width, img_height))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

#Get file name
def get_class(img_path):
    return img_path.split('/')[-1]

#leer el fichero y ponerlo en un diccionario python. Leer imagenes y guardar en un diccionario su path y su nombre.
#comparar para el resultado labelsDict and filesDict
labelsDict = {}
with open('/home/dlcv/DeployedProjects/Jam/dlcv/train/annotation.txt', 'r') as f:
    next(f)
    for line in f:
        (key, val) = line.split()
        labelsDict[key] = val

all_img_paths = glob.glob(os.path.join(path1, '*.jpg'))
np.random.shuffle(all_img_paths)
filesDict = {}
imgs = []
labels = []
#Extract image name, look for it in dictionary, extract label, save. De esta manera tendremos el nombre de la imagen y los labels en el mismo orden
for file in all_img_paths:
    img = preprocess_img(io.imread(file))
    img_name = get_class(file)
    img_name = os.path.splitext(img_name)[0]
    filesDict[file] = img_name
    label = labelsDict[img_name]
    labels.append(label)
    imgs.append(img)

#x and y
X = np.array(imgs, dtype='float32')
#labels as one hot
encoder = LabelBinarizer()
Y = encoder.fit_transform(labels)

#Divide data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)


# build the inception network
model = Inceptionv3(weights = "imagenet", include_top=False, input_shape = (3,img_width, img_height))
print('Model loaded.')

#save the bottleneck features, passing tarrasa buildings
datagen = ImageDataGenerator(rescale=1. / 255)
print(len(X_train))
generator = datagen.flow(
        X_train,
        Y_train,
        batch_size=batch_size,
        shuffle=False)
bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size + 1)
np.save(open('imagenet_features_train.npy', 'w'),
            bottleneck_features_train)
generator = datagen.flow(
        X_test,
        Y_test,
        batch_size=batch_size,
        shuffle=False)
bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size + 1)
np.save(open('imagenet_features_validation.npy', 'w'),
            bottleneck_features_validation)
			
#Freeze layers
for layer in model.layers[:75]:
    layer.trainable = False
# compile the model

#Train the top-model
# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(13, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])
# Save the model according to the conditions
checkpoint = ModelCheckpoint("inceptionv3_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

arch = model.fit(train_data, Y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, Y_test),
                 callbacks=[checkpoint, early])
model.save_weights("imagenet_fc_model.h5")

with open ('transfer_learning_vgg_tarrassa.txt','w') as f:
    f.write(str(arch.history['acc']) + '\n')
    f.write(str(arch.history['loss']) + '\n')
    f.write(str(arch.history['val_acc']) + '\n')
    f.write(str(arch.history['val_loss']))


exit(0)