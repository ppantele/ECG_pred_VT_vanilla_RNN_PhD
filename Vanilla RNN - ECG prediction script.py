'''
Panteleimon Pantelidis, April 2024

Vanilla version of RNN model to detect / predict arrhythmias.
Pilot design and testing. - Main model structure for data format.
'''

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy.io import loadmat


def download_and_preprocess_data(source_url):
    # This function represents data downloading and preprocessing
    # (splitting into 15 sec segments, at 200Hz, ie. 3000 data point-samples).
    # Also, allocate appropriate labels wrt to the temporal position of VT/VF episodes.
    # Outputs: data [nd.array], labels [nd.array]
    pass


def build_model(input_shape):
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        SimpleRNN(128, return_sequences=True),  # First RNN layer, returning sequences for deeper connections
        Dropout(0.25),  # Dropout for regularization
        SimpleRNN(128, return_sequences=True),  # Second RNN layer
        Dropout(0.25),  # Additional dropout
        SimpleRNN(128),  # Third RNN layer, not returning sequences to prepare for output
        Dense(64, activation='relu'),  # Dense layer for non-linear transformation
        Dropout(0.25),  # Dropout in dense layer
        Dense(1, activation='sigmoid')  # Output layer
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Settings
    source_url = "https://physionet.org/files/sddb/1.0.0/"
    window_size = 3000  # 15 sec at 200 Hz
    
    # Load and pre-process (segment, label and arrange) data
    # data, labels = download_and_preprocess_data(source_url)

    # Build the model
    model = build_model((window_size, 2))  # 3000 timesteps, 2 features

    # Train-test split
    # Use a function to split
    # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

    # Train the model
    # model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

    # Evaluate the model
    # loss, accuracy = model.evaluate(test_data, test_labels)
    # print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Optionally save the model
    # model.save('my_rnn_model.h5')

if __name__ == "__main__":
    main()
