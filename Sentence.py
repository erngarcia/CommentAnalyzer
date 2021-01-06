##!/usr/bin/env python
##Title: Comment Analyzer
##Authors: Luis Ernesto Garcia, Jorge Willian Alvarado
##Date: 11/2020
##Modified: 1/2021

import nltk
from nltk.tokenize import word_tokenize
import re
import mysql.connector
from mysql.connector import Error
from pickle import load
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Sentence:
    def __init__(self, text, index, post_id):
        self.text = text
        self.index = index
        self.post_id = post_id

    def contents(self):
        return f"{self.text} in the index {self.index} in the post {self.post_id}"

    def extract_clause_components(self):

        nouns = []
        verbs = []
        adjectives = []
        adverbs = []
        media = []
        hashtags = []
        slogans = []

        regex_tagger_pkl = open('taggers/regex_tagger.pkl', 'rb')
        regex_tagger = load(regex_tagger_pkl)
        regex_tagger_pkl.close()

        fulltext = self.text.split(' ')
        temp_slogan = ""
        for i in fulltext:
            if re.match(r'^http.*', i):
                media.append(i)
                fulltext.remove(i)
            elif re.match(r'^#.*', i):
                hashtags.append(i)
            elif i.isupper() or i.isdigit():
                temp_slogan = temp_slogan + i + ' '
            elif i.islower:  # or fulltext.index(i) == (len(fulltext)-1):
                if len(temp_slogan) > 1:
                    slogans.append(temp_slogan)
                    temp_slogan = ""

        tokens = [token.lower() for token in word_tokenize(' '.join(fulltext))]

        for i in regex_tagger.tag(tokens):
            if re.search('N.+', i[1]):
                nouns.append(i[0])
            elif re.match('V.+', i[1]):
                verbs.append(i[0])
            elif re.match('A.+', i[1]):
                adjectives.append(i[0])

        return len(verbs)
##creates a sentiment analyzer model to determine polarity of the comments.
    def sentiment_analyzer_model(self):
        # best fitting dataset for the purposes of this program.
        dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        encoder = info.features['text'].encoder
        BUFFER_SIZE = 10000
        BATCH_SIZE = 64
        padded_shapes = ([None], ())
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
        test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
        cehckpoint_path = "training/cp.ckpt"
        checkpoint_dir = os.path.dirname(cehckpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(cehckpoint_path, save_weights_only=True, verbose=1)

        # model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

        # train the model
        # history = model.fit(train_dataset, epochs=5, validation_data=test_dataset, validation_steps=30, callbacks=[cp_callback])

        # define the model, activation sigmoid outputs the better performance for law analysis.
        def create_model():
            model = tf.keras.Sequential(
                [tf.keras.layers.Embedding(encoder.vocab_size, 64),
                 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                 tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(1, activation='sigmoid')])
            return model

        model = create_model()
        model.load_weights(cehckpoint_path)

        ##normalizes the size of the training data
        def pad_to_size(vec, size):
            zeros = [0] * (size - len(vec))
            vec.extend(zeros)
            return vec

        # function to determine the sentences sentiment.
        def sample_predict(sentence, pad):
            encoded_sample_pred_text = encoder.encode(sentence)
            if pad:
                encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)

            encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
            predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
            return predictions

        predictions = sample_predict(self.text, pad=True) * 100
        print(sentence+" "+predictions)
        return predictions



class Connection:

    def select_post_messages(self):
        try:
            connection = mysql.connector.connect(
                host=,
                user=,
                passwd=,
                database=
            )
            sql_select = "SELECT id, message FROM posts"
            cursor = connection.cursor()
            cursor.execute(sql_select)
            record = cursor.fetchall()

            return record
        except Error as e:
            message = "Error reading data from database"
            return message


    def insert_post_info(self, sentence):
        tables = ["verb", "noun", "adverb", "adjective"]
        id = 0
        count = 0

        element_count = 0
        for table in tables:
            try:
                connection = mysql.connector.connect(
                    host=,
                    user=,
                    passwd=,
                    database=
                )
                sql_select = "INSERT INTO " + table + "s" + "(post_id," + table + " ,count) VALUES (" + id + "," + \
                             sentence.table()[count] + "," + element_count + ")"
                cursor = connection.cursor()
                cursor.execute(sql_select)
                record = cursor.fetchall()

                message = "records inserted correctly"
                return message
            except Error as e:
                message = "Error reading data from database"
                return message


class Post:
    def __init__(self, post_id, text):
        self.text = text
        self.post_id = post_id

    def parser(self):
        s = re.split('\.\s', self.text)
        return s


def main():
    con1 = Connection()
    post_messages = con1.select_post_messages()
## Generates a plot graph to check relationship between verbs and polarity of the sentence.
    x = []
    y = []
    colours = (0,0,0)
    area = np.pi*3
    for m1 in post_messages:
        id, parrafo = m1

        ps = Post(id, parrafo)
        p = ps.parser()
        for i in range(len(p)):
            sen1 = Sentence(p[i], i, id)
            x.append(sen1.extract_clause_components())
            y.append(sen1.sentiment_analyzer_model())
    plt.scatter(x,y,s=area,c=colours,alpha=0.5)
    plt.title("Relacion entre cantidad de verbos y polaridad del post de facebook")
    plt.xlabel("cantidad de verbos")
    plt.ylabel("polaridad")
    plt.show()
    pass


if __name__ == '__main__':
    main()
