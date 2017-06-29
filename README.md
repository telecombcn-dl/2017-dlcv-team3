# 2017-dlcv-team3
Team 3 in the Deep Learning for Computer Vision Summer School 2017 at UPC TelecomBCN.


## Edges2Faces

This model task is based on pix2pix architecture. You can find all the information in [this](https://github.com/affinelayer/pix2pix-tensorflow.git) repository.  The aim of the project is to use this architecture in order to generate Faces from edges, similar to Edges2Cats or Edges2Shoes.

### Prerequisites
- Tensorflow 1.0.0
- OpenCV

### Recomended
- Linux with Tensorflow GPU edition + cuDNN

### Setup
The first thing shoul be to download the model from the pix2pix repository.

```
git clone https://github.com/affinelayer/pix2pix-tensorflow.git

```

Also, we will need to download the database of faces in the wild. [link](http://vis-www.cs.umass.edu/lfw/)
```
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

```

