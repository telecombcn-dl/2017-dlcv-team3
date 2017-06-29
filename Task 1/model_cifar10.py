from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Hyperparameters
batch_size = 100 # in each iteration, we consider 32 training examples at once
num_epochs = 2 # we iterate 50 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

num_classes = 10

def define_CNN(x_train, kernel_size, pool_size, drop_prob_1, drop_prob_2, hidden_size, num_classes, summary):
    model = Sequential()

    model.add(Conv2D(32, (kernel_size, kernel_size), padding='same', activation='relu', input_shape=x_train.shape[1:], name='conv1'))

    model.add(Conv2D(32, (kernel_size, kernel_size), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='max_pool1'))
    model.add(Dropout(drop_prob_1, name='dropout1'))

    model.add(Conv2D(64, (kernel_size, kernel_size), padding='same', activation='relu', name='conv3'))
    model.add(Conv2D(64, (kernel_size, kernel_size), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='max_pool2'))
    model.add(Dropout(drop_prob_1, name='dropout2'))

    model.add(Conv2D(128, (kernel_size, kernel_size), padding='same', activation='relu', name='conv5'))
    model.add(Conv2D(128, (kernel_size, kernel_size), activation='relu', name='conv6'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), name='max_pool3'))
    model.add(Dropout(drop_prob_1, name='dropout3'))

    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu', name='fc1'))
    model.add(Dropout(drop_prob_2, name='dropout4'))
    model.add(Dense(hidden_size, activation='relu', name='fc2'))
    model.add(Dropout(drop_prob_2, name='dropout5'))
    model.add(Dense(num_classes, activation='softmax', name='fc3'))

    if summary:
        model.summary()

    return model