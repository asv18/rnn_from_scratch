import numpy as np

from utils import softmax

# finally, we have the ouput layer which returns predictions
class OutputLayer:
    # stores predictions of all time steps i.e. internal memory
    # of the network
    states: np.ndarray = None # type: ignore

    # output weight matrix
    V: np.ndarray = None # type: ignore

    # gradient of V during backpropagation
    delta_V: np.ndarray = None # type: ignore

    # output bias
    bias: np.ndarray = None # type: ignore

    # gradient of output bias
    delta_bias: np.ndarray = None # type: ignore

    def __init__(self, size: int, hidden_size: int) -> None:
        self.V = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(size, size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_V = np.zeros_like(self.V)

    # forwadd pass to calculate the weighted output and 
    # probability distribution with softmax
    def predict(self, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        # V @ h => (input_size, h_dimension) @ (h_dimension, 1) = (input_size, 1)
        # (input_size, 1) + (input_size, 1) = (input_size, 1)
        output = self.V @ hidden_state + self.bias
        prediction = softmax(output)
        self.set_state(time_step, prediction)
        return prediction

    # return the output state i.e. prediction value at a given
    # time step
    def get_state(self, time_step: int) -> np.ndarray:
        return self.states[time_step]

    # updating the output state at a time step after forward pass
    # calculation
    def set_state(self, time_step: int, prediction: np.ndarray) -> None:
        self.states[time_step] = prediction

    # compute gradients of V and c
    def calculate_deltas_per_step(
        self,
        expected: np.ndarray,
        hidden_state: np.ndarray,
        time_step: int,
    ) -> np.ndarray:
        # dL_do = dL_dyhat * dyhat_do = derivative of loss function * derivative of softmax
        # dL_do = step.y_hat - expected[step_number]
        delta_output = self.get_state(time_step) - expected  # (input_size, 1)

        # (input_size, 1) @ (1, hidden_size) = (input_size, hidden_size)
        self.delta_V += delta_output @ hidden_state.T

        # dL_dc += dL_do
        self.delta_bias += delta_output
        return self.V.T @ delta_output

    # updating the parameters using the gradient
    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias