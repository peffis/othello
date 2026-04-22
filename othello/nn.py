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

    def evaluate(self, board) -> float:
        X_values = board.toInputVector()
        return float(self.model(X_values, training=False)[0][0])

    def trainable_variables(self):
        return self.model.trainable_variables
