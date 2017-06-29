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




<center>
<iframe src="//www.slideshare.net/slideshow/embed_code/key/5cXl80Fm2c3ksg" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/salgan-visual-saliency-prediction-with-generative-adversarial-networks" title="SalGAN: Visual Saliency Prediction with Generative Adversarial Networks" target="_blank">SalGAN: Visual Saliency Prediction with Generative Adversarial Networks</a> </strong> from <strong><a target="_blank" href="//www.slideshare.net/xavigiro">Xavier Giro</a></strong> </div>
</center>

## Publication

Find the extended pre-print version of our work on [arXiv](https://arxiv.org/abs/1701.01081). The shorter extended abstract presented as spotlight in the [CVPR 2017 Scene Understanding Workshop (SUNw)](http://sunw.csail.mit.edu/) is available [here](https://github.com/imatge-upc/saliency-salgan-2017/raw/master/papers/sunw-2017-abstract.pdf).

![Image of the paper](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/figs/thumbnails.jpg)

Please cite with the following Bibtex code:

```
@InProceedings{Pan_2017_SalGAN,
author = {Pan, Junting and Canton, Cristian and McGuinness, Kevin and O'Connor, Noel E. and Torres, Jordi and Sayrol, Elisa and Giro-i-Nieto, Xavier and},
title = {SalGAN: Visual Saliency Prediction with Generative Adversarial Networks},
booktitle = {arXiv},
month = {January},
year = {2017}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Cristian Canton, Kevin McGuinness, Noel E. O'Connor, Jordi Torres, Elisa Sayrol and Xavier Giro-i-Nieto. "SalGAN: Visual Saliency Prediction with Generative Adversarial Networks." arXiv. 2017.*



## Models

The SalGAN presented in our work can be downloaded from the links provided below the figure:

SalGAN Architecture
![architecture-fig]

* [[SalGAN Generator Model (127 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz)
* [[SalGAN Discriminator (3.4 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz)

[architecture-fig]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D "SALGAN architecture"
[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt

## Visual Results

![Qualitative saliency predictions](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/qualitative.jpg?token=AFOjyaO0uT7l7qGzV7IyrcSgi8ieeayTks5Yc4s2wA%3D%3D)


## Datasets

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [SALICON](http://salicon.net/).

### Test
Two different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).


## Software frameworks

Our paper presents two convolutional neural networks, one correspends to the Generator (Saliency Prediction Network) and the another is the Discriminator for the adversarial training. To compute saliency maps only the Generator is needed.

### SalGAN on Lasagne

SalGAN is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
```
pip install -r https://github.com/imatge-upc/saliency-salgan-2017/blob/junting/requirements.txt
```

### Usage

To train our model from scrath you need to run the following command:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 02-train.py
```
In order to run the test script to predict saliency maps, you can run the following command after specifying the path to you images and the path to the output saliency maps:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 03-predict.py
```
With the provided model weights you should obtain the follwing result:

| ![Image Stimuli]  | ![Saliency Map]  |
|:-:|:-:|

[Image Stimuli]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/images/i112.jpg
[Saliency Map]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/saliency/i112.jpg

Download the pretrained VGG-16 weights from: [vgg16.pkl](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl)


## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the projects [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application) and [Malegra TEC2016-75976-R](https://imatge.upc.edu/web/projects/malegra-multimodal-signal-processing-and-machine-learning-graphs), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289. |  ![logo-ireland] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"
[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/saliency-salgan-2017/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.

<!---
Javascript code to enable Google Analytics
-->

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-7678045-13', 'auto');
  ga('send', 'pageview');

</script>