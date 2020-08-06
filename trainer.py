import numpy as np

class LayerwiseTrainer():
    """
    This is a flexible class for training specific layers of deep neural-nets in an online manner.
    """
    def __init__(self, model):
        self.m = model
        self.refs = {}

    def compile(self, ignore=None, *args, **kwargs):
        """
        Performs compilation of the model for the layer-wise training.

        ignore: list of layers to ignore (will perform substring matching without case)
        *args|**kwargs: parameters to be fed into keras.Model.compile
        """
        if not ignore:
            ignore = []
        self.ignore = ignore
        self.m.compile(*args, **kwargs)
        for i in range(len(self.m.layers)):
            model_ref = Model(self.m.layers[0].output, self.m.layers[-1].output)
            layer_name = self.m.layers[i].name.lower()
            if any([nm.lower() in layer_name for nm in ignore]):
                continue
            for j in range(len(self.m.layers)):
                if i == j:
                    model_ref.layers[j].trainable = True
                else:
                    model_ref.layers[j].trainable = False
            model_ref.compile(*args, **kwargs)
            self.refs.update({layer_name:model_ref})

    def fit_by_batch(self, x, y, epochs, batch_size, validation_data=None, verbose=None, *args, **kwargs):
        """
        Fits the model based on layerwise updates.

        x: input data
        y: output data
        epochs: number of outer iterations
        batch_size: number of examples to sample from `x`
        validation_data: an iterable of input and output data ('verbose' must be set to see values)
        """
        num_examples = x.shape[0]
        num_batches = num_examples // batch_size
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch: {epoch}")
                if validation_data:
                    print(self.m.evaluate(*validation_data))
            for batch in range(num_batches):
                for layer in self.refs.keys():
                    idx = np.random.randint(0, num_examples, batch_size)
                    x_batch = x[idx]
                    y_batch = y[idx]
                    self.refs[layer].train_on_batch(x_batch, y_batch)
