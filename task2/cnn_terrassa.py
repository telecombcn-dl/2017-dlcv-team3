import os
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from sklearn.preprocessing import LabelBinarizer
from numpy import *
from skimage import color, exposure, transform, io
from sklearn.cross_validation import train_test_split
import glob
from keras import backend as K
K.set_image_data_format('channels_first')

# dimensions of our images.
img_width, img_height = 200, 200

top_model_weights_path = 'tb_model.h5'
train_data_dir = 'train'
validation_data_dir = 'val'
nb_total_samples = 894
nb_train_samples = 713
nb_validation_samples = 181

epochs = 32
batch_size = 50
num_classes = 13

path1 = '/home/dlcv/DeployedProjects/Jam/dlcv/train/images'
listing = os.listdir(path1)
num_samples=nb_total_samples

#preprocess
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
#imlist = os.listdir(path2)
#im1 = array(Image.open('input_data_resized' + '\\'+ imlist[0])) # open one image to get size
#m,n = im1.shape[0:2] # get the size of the images
#imnbr = len(imlist) # get the number of images

#x and y
X = np.array(imgs, dtype='float32')
#labels as one hot
encoder = LabelBinarizer()
Y = encoder.fit_transform(labels)

#Divide data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
#save data
np.save(open('X_train.npy', 'w'), X_train)
np.save(open('X_test.npy', 'w'), X_test)
np.save(open('Y_train.npy', 'w'), Y_train)
np.save(open('Y_test.npy', 'w'), Y_test)

#Model definition
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(3, img_height, img_width)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#train
# initiate RMSprop optimizer
opt = rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation.')
    arch = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X)

    # Fit the model on the batches generated by datagen.flow().
    arch = model.fit_generator(datagen.flow(X, Y,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))



with open ('tarrasa_own_cnn_32epochs.txt','w') as f:
    f.write(str(arch.history['acc']) + '\n')
    f.write(str(arch.history['loss']) + '\n')
    f.write(str(arch.history['val_acc']) + '\n')
    f.write(str(arch.history['val_loss']))
print "end"
exit(0)

