Deep Structure Transfer Machine
=============

This code was only used for CIFAR image classification experiments with Deep Structure Transfer Machine based on Tensorflow. Other experiments are done based on Caffe, the source code will be released soon.


# Installation

The code is developed using Tensorflow 1.2.

For simplicity, TensorLayer is used for quitck loading the Cifar data. Noted that in order to fully reproduce the performance reported in the paper, please follow Wide Residual Networks [1] for data preprocessing.

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

# References
- Zagoruyko, Sergey, and N. Komodakis. "Wide Residual Networks." (2016).

