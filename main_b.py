# This is the main python file for the project

# Tensorflow library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Other helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Task B

# Load the dataset
train_dataset = pd.read_csv('./Data/Training/B/subtaskB_data_all.csv')
train_answers = pd.read_csv('./Data/Training/B/subtaskB_answers_all.csv')

# Remove the id column since it is irrelevant for us
train_dataset.pop('id')
train_dataset.pop('FalseSent')
train_answers.pop('0')

# Load the the sentences
sentences = []
counter = 0
for item in train_dataset.loc:
    sentences.append(item['OptionA'])
    sentences.append(item['OptionB'])
    sentences.append(item['OptionC'])
    counter = counter + 1
    if counter > 9999:
        break

# Load the answers
load_answers = [10000]
load_answers[0] = 'B'
counter = 0
for item in train_answers.loc:
    load_answers.append(item['B'])
    counter = counter + 1
    if counter > 9998:
        break

# Make an answer for every sentence in sentences[]
answers = []
index = 0
for item in range(0, len(load_answers), 1):
    if load_answers[item] == 'A':
        answers.insert(index, 1)
        answers.insert(index+1, 0)
        answers.insert(index+2, 0)
    elif load_answers[item] == 'B':
        answers.insert(index, 0)
        answers.insert(index + 1, 1)
        answers.insert(index + 2, 0)
    elif load_answers[item] == 'C':
        answers.insert(index, 0)
        answers.insert(index + 1, 0)
        answers.insert(index + 2, 1)
    index += 4

# Make a training dataset and a testing dataset from the sentences above
train_sentences = sentences[0:25000]
test_sentences = sentences[25000:]

# Make answers for the sentences
train_sentences_answers = answers[0:25000]
test_sentences_answers = answers[25000:]

# Hyperparameters
vocab_size = 7000
embedding_dim = 16
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"

# Tokenize each word
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(train_sentences)
testing_sequences = tokenizer.texts_to_sequences(test_sentences)

# Add padding ot each sentence to be the same length
padded_training = pad_sequences(training_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
padded_testing = pad_sequences(testing_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Convert the answer lists to arrays
train_answers_array = np.asarray(train_sentences_answers)
test_answers_array = np.asarray(test_sentences_answers)

# Define loss funtion and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_training, train_answers_array, epochs=30)

# Test the model
test_loss, test_acc = model.evaluate(padded_testing, test_answers_array, verbose=1)

# Print the model's accuracy
print('Model\'s accuracy:', test_acc)