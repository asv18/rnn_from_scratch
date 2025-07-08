import string
import numpy as np
import sys
import tensorflow as tf

from utils import string_to_one_hot
from vanilla_rnn import VanillaRNN

# let us define our sample training data below:
inputs = np.array([
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
    ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
    ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
    ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
])

expected = np.array([
    ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"], 
    ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
    ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
])

if len(sys.argv) == 2:
    if sys.argv[1] == "vanilla":
        one_hot_inputs = string_to_one_hot(inputs)
        one_hot_expected = string_to_one_hot(expected)

        # Forward pass through time, no gradient clipping yet so there will be gradient exploding problem
        # https://stackoverflow.com/a/33980220
        # https://stackoverflow.com/a/72494516
        rnn = VanillaRNN(vocab_size=len(string.ascii_uppercase), hidden_size=128, alpha=0.0001)
        rnn.train(one_hot_inputs, one_hot_expected, epochs=10)

        print()

        new_inputs = np.array([["B", "C", "D"]])
        one_hot_new_inputs = string_to_one_hot(new_inputs)
        for idx in range(0, len(one_hot_new_inputs)):
            predictions = rnn.feed_forward(one_hot_new_inputs[idx])
            output = np.argmax(predictions.states[-1])
            print(f"input: {new_inputs[idx]}")
            print(f"output index: {output}") # index of the one-hot value of prediction
            print(f"output character: {string.ascii_uppercase[output]}") # mapping one hot to character
    elif sys.argv[1] == "tf":
        # Encode strings to int indexes
        input_encoded = np.vectorize(string.ascii_uppercase.index)(inputs)
        input_encoded = input_encoded.astype(np.float32)
        one_hot_inputs = tf.keras.utils.to_categorical(input_encoded)

        expected_encoded = np.vectorize(string.ascii_uppercase.index)(expected)
        expected_encoded = expected_encoded.astype(np.float32)
        one_hot_expected = tf.keras.utils.to_categorical(expected_encoded)

        rnn = tf.keras.layers.SimpleRNN(128, return_sequences=True)

        model = tf.keras.Sequential(
            [
                rnn,
                tf.keras.layers.Dense(len(string.ascii_uppercase)),
            ]
        )

        model.compile(loss="categorical_crossentropy", optimizer="adam")

        model.fit(one_hot_inputs, one_hot_expected, epochs=10)

        new_inputs = np.array([["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"]])
        new_inputs_encoded = np.vectorize(string.ascii_uppercase.index)(new_inputs)
        new_inputs_encoded = new_inputs_encoded.astype(np.float32)
        new_inputs_encoded = tf.keras.utils.to_categorical(new_inputs_encoded)
        
        # Make prediction
        prediction = model.predict(new_inputs_encoded)

        # Get prediction of last time step
        prediction = np.argmax(prediction[0][-1])
        print(prediction)
        print(string.ascii_uppercase[prediction])
else:
    print("err - invalid args. provide either \"vanilla\" or \"tf\"")