Deep Transfer Machine
=============

This code was used for cifar image classification experiments with Deep Transfer Machine.


# Installation

The code is developed using Tensorflow. We tested our code using Tensorflow 1.2.

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
Note that the weight for manifold loss is set to be 0 by default, to train the network using manifold loss, run:

```
python manifold.py --manifold_weight sigma
```
Several hyper-parameter can be set by add  --manifold_weigh, --learning_rate, --decay_step and etc.

An example log file is also attached for reference.

