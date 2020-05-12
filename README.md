# Music Style Transfer

> Authors: Henrik Høiness (henrhoi), Axel Harstad (axeloh) and Marius Landsverk Cervera (mariusmcl)

Deep learning models trained on visual data have changed the field of computer vision and is now finding more and more ways into consumer products and business applications.

Humans however are heavily dependent on audio data to navigate their lives, and being able to create powerful models for raw audio data has numerous applications ranging from the arts to new business opportunities. 
In this work we expand upon recent architectural advances (*Aaron van den Oord et. al.: WaveNet: A Generative Model for Raw Audio (2016)* ) and propose a WaveNet-like autoencoder with a shared encoder and multiple decoders to perform style transfer between multiple musical instruments.

Our method consists in training multiple decoders, one for each domain, together with a shared encoder. In order to enforce a disentangled latent representation, a domain classifier is trained to classify the latent representations' domain.

The architecture is heavily inspired by *Noem Mor et al.: A Universal Music Translation Network (2018)* ([repo](https://github.com/facebookresearch/music-translation)).

All samples in this repo are from a WaveNet Autoencoder trained 4 days on two Tesla V100 with two decoders for the two domains <i>Bach Solo Cello</i> and <i>Beethoven Solo Piano</i>. 

You can listen to the samples and see the architecture used [here](https://henrikhoiness.com/2020/05/10/audio-style-transfer/).

### Dataset 

Dataset used in experiments is `Musicnet`, which can be found [here](https://homes.cs.washington.edu/~thickstn/musicnet.html)

To preprocess and extract domains from the dataset, run the scripts below in the following order:

1. `$ python preprocessing/seperate_domains.py`
2. `$ python preprocessing/train_test_val_split.py`
3. `$ python preprocessing/preprocess.py`


### Training

**Pre-trained models can be downloaded [here](https://drive.google.com/file/d/1-PVqPnQEO3fBf5H166oGK7PggzqbaBVD/view?usp=sharing) and place it under `checkpoints/trained_models`**

To train on a single GPU-node run:

 `$ python train_music_translation.py`



### Generate

In order to generate samples from `dataset/samples/input`, run:

 `$ python generate_style_and_instrument_transfer.py`