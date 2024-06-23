# Convolutional Neural Network (CNN) modules from scratch

## About modularCNN

The main purpose of this project is to efficiently facilitate the construction and implementation of CNN architectures from scratch, WITHOUT the use of existing deep learning packages such as keras or pytorch. With this library, you can experiment with building various neural network architectures for image classification, from a simple one-hidden-layer perceptron to a deep network of multiple Convolutional and Pooling layers, with the appropriate activation functions and weights initialisation strategies. Training performance is fairly efficient due to the use of fully vectorised numpy operations at the layer level, though still a little slower compared to established libraries like keras.

## Instalation and Use Guide
To work with this library on your local machine, simply:
1. Clone the repository,

```
!git clone https://github.com/nam-ngh/modularCNN.git
```

2. Import modules:

```
from lib import layer, network
```

3. Now you can easily build your own neural networks, for example:<br>
 - Defining your model with the Net class:

```
model = network.Net()
```

 - adding in layers:

```
model.add(layer.Convolutional(input_shape=(32,32,3),filters=16,filter_size=3,stride=1,pad=1))
# e.g.: a Convolutional layer with 16 3x3 filters, stride of 1 and padding of 1 pixel
```

 - get a summary of the architecture:

```
model.summary()
```

 - and train the model:

```
model.train(x_train, y_train, epochs=30, learn_rate=0.00001, val_size=0.1)
# split 10% of training data for validation
```
A simple build and train example with cifar10 is included!

## Important Notes
- The current model and layers are only compatible with square images: Input feature sets x_train, x_test must be provided in shape (n,x,x,c) where n = number of sample, x = height and width of the image, and c = number of channels (c=1 for b&w, c=3 for RGB)
- input_shape must be specified for all layers upon adding to the network, in the form of (x,x,c), with the exception of Activation layer where this is not needed and Dense layer, where the number of neurons in and neurons out are required instead.
- Ouput size of Convolutional and MaxPooling layer can be determined as follows: o = (x - filter_size + 2*pad)/stride + 1. Please make sure this is a whole number so that convolutions are complete and free of errors
- It is recommmended that you determine the output shape of the previous layer first before adding the next, to make sure that the shapes don't mismatch. If you are unsure you can run the model summary function each time you add a layer to check the layers added so far
- Currently for MaxPooling layer, pool_size and stride can only be set as the same number under pool_size, i.e. pooling regions can't overlap