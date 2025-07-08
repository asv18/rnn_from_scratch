# A class to wrap everything together
from typing import List
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer

import numpy as np

class VanillaRNN:
    hidden_layer: HiddenLayer
    output_layer: OutputLayer
    alpha: float  # learning rate
    input_layer: InputLayer = None # type: ignore

    def __init__(self, vocab_size: int, hidden_size: int, alpha: float) -> None:
        self.hidden_layer = HiddenLayer(vocab_size, hidden_size)
        self.output_layer = OutputLayer(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.alpha = alpha

    # forward pass to iterate through the input sequence
    # and coordinate functions from other layers to compute predictions
    # at each time step
    def feed_forward(self, inputs: np.ndarray) -> OutputLayer:
        self.input_layer = InputLayer(inputs, self.hidden_size)
        for step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input, step)
            self.output_layer.predict(activation, step)
        return self.output_layer

    # backward pass to iterate through the sequence but in reverse order
    # - calculates and updates the gradients for all parameters and
    # eventually updates weights and biases
    def backpropagation(self, expected: np.ndarray) -> None:
        for step_number in reversed(range(len(expected))):
            delta_output = self.output_layer.calculate_deltas_per_step(
                expected[step_number],
                self.hidden_layer.get_hidden_state(step_number),
                step_number,
            )
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(
                step_number, delta_output
            )
            self.input_layer.calculate_deltas_per_step(step_number, delta_weighted_sum)

        self.output_layer.update_weights_and_bias(self.alpha)
        self.hidden_layer.update_weights_and_bias(self.alpha)
        self.input_layer.update_weights_and_bias(self.alpha)

    # cross-entropy loss function
    def loss(self, y_hat: List[np.ndarray], y: List[np.ndarray]) -> float:
        """
        Cross-entropy loss function - Calculating difference between 2 probability distributions.
        First, calculate cross-entropy loss for each time step with np.sum, which returns a numpy array
        Then, sum across individual losses of all time steps with sum() to get a scalar value.
        :param y_hat: predicted value
        :param y: expected value - true label
        :return: total loss
        """
        return sum(-np.sum(y[i] * np.log(y_hat[i]) for i in range(len(y)))) # type: ignore

    # run through the number of epochs given and iterate each input
    # sequence with both forward and backward pass and calculate the loss
    def train(self, inputs: np.ndarray, expected: np.ndarray, epochs: int) -> None:
        for epoch in range(epochs):
            print(f"epoch={epoch}")
            for idx, input in enumerate(inputs):
                y_hats = self.feed_forward(input)
                self.backpropagation(expected[idx])
                print(
                    f"Loss round: {self.loss([y for y in y_hats.states], expected[idx])}"
                )