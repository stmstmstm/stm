Deep Structure Transfer Machine
=============

This code was used for cifar image classification experiments with Deep Structure Transfer Machine.


# Installation

The code is developed using Tensorflow.

For simplicity, TensorLayer is used for quitck loading the cifar data. Note that in order to fully reproduce the performance reported in the paper, data augmentations like random flipping and translation are needed.

To install TensorLayer, you can simply run:

```
pip install tensorlayer
```

# Training

You can train the network by running:

```
python manifold.py --gpu 0 --log_dir log1
```
The weight for manifold loss is set to be 0.2 by default, to train the network using manifold loss, run:

```
python manifold.py --manifold_weight sigma
```
Other hyper-parameters can be set by adding  --manifold_weigh, --learning_rate, --decay_step and etc.

An example log file is also attached for reference.

