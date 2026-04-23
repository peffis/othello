import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer


def get_model(n_inputs, n_outputs, n_hidden):
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,), name='input_layer'))
    model.add(Dense(n_hidden, kernel_initializer='he_uniform', activation='sigmoid', name='hidden_layer'))
    model.add(Dense(n_outputs, name='output_layer', activation='tanh'))
    return model


def get_conv_model(board_shape=(8, 8, 2), n_filters=32, n_hidden=64):
    """2-layer Conv2D value net for 8x8 board input.

    Input(8,8,2) -> Conv2D(n_filters,3,same,relu) x2 -> Flatten
                 -> Dense(n_hidden,relu) -> Dense(1,tanh)
    """
    model = Sequential()
    model.add(InputLayer(shape=board_shape, name='input_layer'))
    model.add(Conv2D(n_filters, 3, padding='same', activation='relu', name='conv1'))
    model.add(Conv2D(n_filters, 3, padding='same', activation='relu', name='conv2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(n_hidden, kernel_initializer='he_uniform', activation='relu', name='hidden_layer'))
    model.add(Dense(1, activation='tanh', name='output_layer'))
    return model


class NN:
    def __init__(self, params=None, *, model=None):
        """Wrap a Keras model with tf.function-compiled forward and
        forward-and-grad entry points. Accepts either:
          - params: dict kwargs for get_model(...), for the classic MLP
          - model=: a pre-built Keras model (e.g. from get_conv_model)
        """
        if model is None:
            if params is None:
                raise ValueError("NN needs either params or model=")
            model = get_model(**params)
        self.model = model

        # tf.function-wrapped forward pass for move selection. Varying batch
        # sizes across candidate evaluations would normally cause retracing;
        # reduce_retracing=True collapses them to a single generic signature.
        @tf.function(reduce_retracing=True)
        def _forward(x):
            return self.model(x, training=False)

        # tf.function-wrapped forward + gradient for the TD(λ) update.
        # Lives in the same graph as _forward so the shared Keras layers get
        # compiled once and reused.
        @tf.function(reduce_retracing=True)
        def _forward_and_grad(x):
            with tf.GradientTape() as tape:
                p = self.model(x, training=False)
            grads = tape.gradient(p, self.model.trainable_variables)
            return p, grads

        self._forward = _forward
        self._forward_and_grad = _forward_and_grad

    def evaluate(self, board) -> float:
        X_values = board.toInputVector()
        return float(self._forward(X_values)[0][0])

    def trainable_variables(self):
        return self.model.trainable_variables
