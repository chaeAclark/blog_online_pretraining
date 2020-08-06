<p align="center">
<img width=300 height=350 src="logo.png">
</p>

<h1 align="center">
Online Pre-Training Framework
</h1>

This is a flexible class for training specific layers of deep neural-nets in an online manner. This is accompanied with a blog post [HERE](https://www.twosixlabs.com/blog/). The pre-training refers to the greedy layer-wise training that the framework uses. The method is online in the sense that the model can be pre-trained with new data as it arrives.

## Installing the class from source
```bash
git clone https://github.com/chaeAclark/blog_online_pretraining.git
cd blog_online_pretraining
pip install .
```

## Example
```python
from trainer import LayerwiseTrainer

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Input, Model
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten

# Load data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train - 255./2
x_test = x_test - 255./2
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
D = x_train.shape[1:]
d = y_train.shape[1]

# Create model
x_in = Input(shape=D)
x = Conv2D(filters=32, kernel_size=(3,3), strides=1, activation="relu")(x_in)
x = Conv2D(filters=32, kernel_size=(3,3), strides=1, activation="relu")(x)
x = MaxPool2D(2)(x)
x = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation="relu")(x)
x = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation="relu")(x)
x = MaxPool2D(2)(x)
x = Flatten()(x)
x = Dense(units=128, activation="relu")(x)
x = Dropout(.5)(x)
x = Dense(units=128, activation="relu")(x)
x = Dropout(.5)(x)
x_out = Dense(units=d, activation="softmax", name="output")(x)
model = Model(x_in, x_out, name="Model-Example")

# Create trainer
trainer = LayerwiseTrainer(model)

# Train model
params = {"loss":"categorical_crossentropy", "metrics":["accuracy"], "optimizer":"nadam"}
trainer.compile(**params)
trainer.fit_by_batch(x_train, y_train, epochs=12, batch_size=256, validation_data=(x_test,y_test), verbose=1)

# Output results
model.evaluate(x_test, y_test)
```

## Additional Features
### Ignoring layers
As added functionality, you can ignore layers that you either want trainable (either because they have few/no parameters, or because they are integral to the model). When passing the model into the trainer simnply specify the layer names that should be ignored.
```python
trainer = LayerwiseTrainer(model=model)
trainer.compile(ignore=["dropout","flatten","pool","output"], **params)
trainer.fit_by_batch(x_train, y_train, epochs=12, batch_size=256, validation_data=(x_test,y_test), verbose=1)
```
