# '''Visualization of the filters of VGG16, via gradient ascent in input space.
# This script can run on CPU in a few minutes (with the TensorFlow backend).

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print(model.summary())
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'block1_conv1'
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
plt.savefig('visualization_vgg16_block1_conv1')

