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

Now crop the images in order to have no more than the face in the image.  This will generate a folder with all the images cropped (lfw-deepfunneled_croped)

```
python crop_images.py
```

Then, resize the images so they have the correct size for the model.

```
python pix2pix-tensorflow/tools/process.py \
  --input_dir lfw-deepfunneled_croped \
  --operation resize \
  --output_dir faces_resized
```

Now, the edges will be computed:

```
python generate_edges.py
```

The dataset that will be fed to the model is generated from both the original (cropped and resized) photos and the edges of these photos (Check pix2pix repo for more detail).  Also, generate train and validation partitions from the dataset.

```
python pix2pix-tensorflow/tools/process.py \
  --input_dir faces_resized \
  --b_dir faces_edges \
  --operation combine \
  --output_dir faces_combined

  python pix2pix-tensorflow/tools/split.py \
  --dir faces_combined
```


![model][model-photo]

[model-photo]: ./Edges2Faces/images/model.png "Model Photo"

Finally, in order to train and test the model:

```
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python pix2pix-tensorflow/pix2pix.py \
  --mode train \
  --output_dir faces_train \
  --max_epochs 200 \
  --input_dir faces_combined/train \
  --which_direction BtoA
# test the model
python pix2pix-tensorflow/pix2pix.py \
  --mode test \
  --output_dir faces_test \
  --input_dir faces_combined/val \
  --checkpoint faces_train
```



