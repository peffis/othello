import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, InputLayer


def get_model(n_inputs, n_outputs, n_hidden):
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,), name='input_layer'))
    model.add(Dense(n_hidden, kernel_initializer='he_uniform', activation='sigmoid', name='hidden_layer'))
    model.add(Dense(n_outputs, name='output_layer', activation='tanh'))
    return model


class NN:
    def __init__(self, params):
        self.model = get_model(**params)

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
