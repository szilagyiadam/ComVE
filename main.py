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

# Task A

# Load the dataset
train_dataset = pd.read_csv('./Data/Training/A/subtaskA_data_all.csv')
train_answers = pd.read_csv('./Data/Training/A/subtaskA_answers_all.csv')

# Remove the id column since it is irrelevant for us
train_dataset.pop('id')
train_answers.pop('0')

# Load the the sentences
sentences = []
counter = 0
for item in train_dataset.loc:
    sentences.append(item['sent0'])
    sentences.append(item['sent1'])
    counter = counter + 1
    if counter > 9999:
        break

# Load the answers
load_answers = [10000]
load_answers[0] = 0
counter = 0
for item in train_answers.loc:
    load_answers.append(item['0.1'])
    counter = counter + 1
    if counter > 9998:
        break

# Make an answer for every sentence in sentences[]
answers = []
index = 0
for item in range(0, len(load_answers), 1):
    answers.insert(index, load_answers[item])
    if load_answers[item] == 0:
        answers.insert(index+1, 1)
    elif load_answers[item] == 1:
        answers.insert(index+1, 0)
    index += 2

# Make a training dataset and a testing dataset from the sentences above
train_sentences = sentences[0:16000]
test_sentences = sentences[16000:]

# Make answers for the sentences
train_sentences_answers = answers[0:16000]
test_sentences_answers = answers[16000:]

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
# print(len(word_index))

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

# Make predictions
sentence_predictions = []
good_sentence = input("Type one good sentence: ")
bad_sentence = input("Type one bad sentence which is similar: ")

sentence_predictions.append(good_sentence)
sentence_predictions.append(bad_sentence)

token_sentences = tokenizer.texts_to_sequences(sentence_predictions)
padded_sentences = pad_sequences(token_sentences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

prediction = model.predict(padded_sentences)

if prediction[0] > prediction[1]:
    print("This is the good sentence (maybe): " + good_sentence)
else:
    print("This is the good sentence (maybe): " + bad_sentence)
