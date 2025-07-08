import numpy as np

# the following is our hidden layer, which does a lot of the heavy 
# lifting by calculating deltas and updating weights and biases
class HiddenLayer:
    # stores activation of all time steps i.e. the internal memory
    # of the network
    states: np.ndarray = None # type: ignore

    # the recurrent weight matrix
    W: np.ndarray = None # type: ignore

    # gradient of W during backpropagation
    delta_W: np.ndarray = None # type: ignore

    # the bias offset
    bias: np.ndarray = None # type: ignore

    # gradient of the bias
    delta_bias: np.ndarray = None # type: ignore

    # stores the derivative of the next step's loss function wrt
    # current activation i.e. gradient of activation at the current
    # time step
    next_delta_activation: np.ndarray = None # type: ignore

    def __init__(self, vocab_size: int, size: int) -> None:
        self.W = np.random.uniform(low=0, high=1, size=(size, size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(vocab_size, size, 1))
        self.next_delta_activation = np.zeros(shape=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_W = np.zeros_like(self.W)

    # return the hidden state value at a given time step
    # if time step is less than 0, default to the zero matrix
    def get_hidden_state(self, time_step: int) -> np.ndarray:
        # If starting out at the beginning of the sequence, a[t-1] will return zeros
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    # update the state at a time step after forward pass calculation
    def set_hidden_state(self, time_step: int, hidden_state: np.ndarray) -> None:
        self.states[time_step] = hidden_state

    # forward pass calculation
    def activate(self, weighted_input: np.ndarray, time_step: int) -> np.ndarray:
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        # W @ h_prev => (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension, 1)
        weighted_hidden_state = self.W @ previous_hidden_state
        # (h_dimension, 1) + (h_dimension, 1) + (h_dimension, 1) = (h_dimension, 1)
        weighted_sum = weighted_input + weighted_hidden_state + self.bias
        activation = np.tanh(weighted_sum)  # (h_dimension, 1)
        self.set_hidden_state(time_step, activation)
        return activation

    # compute gradients of W and b
    def calculate_deltas_per_step(
        self, time_step: int, delta_output: np.ndarray
    ) -> np.ndarray:
        # (h_dimension, 1) + (h_dimension, 1) = (h_dimension, 1)
        delta_activation = delta_output + self.next_delta_activation
        # (h_dimension, 1) * scalar = (h_dimension, 1)
        delta_weighted_sum = delta_activation * (
            1 - self.get_hidden_state(time_step) ** 2
        )
        # (h_dimension, h_dimension) @ (h_dimension, 1) = (h_dimension, 1)
        self.next_delta_activation = self.W.T @ delta_weighted_sum

        # (h_dimension, 1) @ (1, h_dimension) = (h_dimension, h_dimension)
        self.delta_W += delta_weighted_sum @ self.get_hidden_state(time_step - 1).T

        # derivative of hidden bias is the same as dL_ds
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum

    # update the parameters using the gradient
    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias