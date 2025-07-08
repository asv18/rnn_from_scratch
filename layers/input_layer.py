import numpy as np

# the following is our input layer, which takes inputs
# and calculates weighted sums, deltas, and updates weights and biases
class InputLayer:
    # sequential data in the form of numpy arrays
    inputs: np.ndarray

    # the weight matrix connecting input to the hidden layer
    U: np.ndarray = None # type: ignore

    # the gradient calculated during Backpropagation through time
    delta_U: np.ndarray = None # type: ignore

    def __init__(self, inputs: np.ndarray, hidden_size: int) -> None:
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(inputs[0])))
        self.delta_U = np.zeros_like(self.U)

    # returns the one-hot encoded vector of the character at a given time step
    def get_input(self, time_step: int) -> np.ndarray:
        return self.inputs[time_step]

    # returns the result of U*x[t] to be used in the weighted sum formula
    def weighted_sum(self, time_step: int) -> np.ndarray:
        return self.U @ self.get_input(time_step)

    # calculate the gradient of U at the given time step
    def calculate_deltas_per_step(
        self, time_step: int, delta_weighted_sum: np.ndarray
    ) -> None:
        # (h_dimension, 1) @ (1, input_size) = (h_dimension, input_size)
        self.delta_U += delta_weighted_sum @ self.get_input(time_step).T

    # update the parameters using the gradient
    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.U -= learning_rate * self.delta_U