
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from vis.utils import utils
#from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters

import model_cifar10 as mynet

import keras
from keras.datasets import cifar10 

kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons
num_classes=10
filepath='weights.cifar10.hdf5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = mynet.define_CNN(x_train, kernel_size, pool_size, drop_prob_1, drop_prob_2, hidden_size, num_classes, 1)
model.load_weights(filepath, by_name=True)

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'conv1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in filters:
    img = visualize_activation(model, layer_idx, filter_indices=idx) 
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)

print(len(vis_images))
# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=8)    
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.savefig('visualization_mynet_conv1')

