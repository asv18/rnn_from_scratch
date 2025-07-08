import string
import numpy as np

# the softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# the following utility function converts the
# list of strings into a list of one-hot encoded vectors
def string_to_one_hot(inputs: np.ndarray) -> np.ndarray:
    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_to_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        one_hot_inputs.append(one_hot_list)

    return np.array(one_hot_inputs)
# each input sequence will have a shape of (26, 26, 1)
# thus, each item matches its index in the alphabet