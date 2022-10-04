from formatter import test
from models import lstm_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from losses import WeightedCategoricalCrossentropy
import sys
import os
import argparse
from typing import List
from collections import Counter

new_data = pd.read_csv('../data/processed/english_luganda2.csv')


def load_data(folds:List[int]):


    train_data=new_data[new_data.folds.isin(folds[:-1])].reset_index(drop=True)
    # train_data=shuffle(train_data).reset_index().drop(columns='index')
    val_data = new_data[new_data.folds==folds[-1]].reset_index().drop(columns='index')
    test_data = new_data[new_data.folds==4].reset_index().drop(columns='index')

    unique_word_count = Counter()
    for i in train_data.text.values:
        for word in i.split():
            unique_word_count[word]+=1

    tokenizer = Tokenizer(num_words=len(unique_word_count))
    tokenizer.fit_on_texts(train_data.text)

    max_length=8
    train_sequences=tokenizer.texts_to_sequences(train_data.text)
    train_padded_sequences = pad_sequences(train_sequences,maxlen=max_length,padding='post',truncating='post')

    val_sequences=tokenizer.texts_to_sequences(val_data.text)
    val_padded_sequences = pad_sequences(val_sequences,maxlen=max_length,padding='post',truncating='post')

    test_sequences=tokenizer.texts_to_sequences(test_data.text)
    test_padded_sequences = pad_sequences(test_sequences,maxlen=max_length,padding='post',truncating='post')
    
    return train_padded_sequences, val_padded_sequences,test_padded_sequences




cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath='/home/g007markphillip/language_classification2/weights_bi2/cp-{epoch:04d}.ckpt',
                    save_weights_only=True,
                    save_freq='epoch')

def run():

    w_array = np.ones([2,2])
    w_array[1,0]=30

    model = lstm_model(embed_dim=32,
                    max_length=8,
                    count=len(unique_word_count),
    #                    vectorize_layer=vectorize_layer
                    )
    optimizer = tf.keras.optimizers.Adam(1e-5)

    loss = WeightedCategoricalCrossentropy(w_array)

    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    # model.build((None,max_length))
    # model.summary()
    # model.fit(tf.convert_to_tensor(train_padded_sequences), train_data[['english','luganda']].values,
    #       validation_data=(val_padded_sequences ,val_data[['english','luganda']].values),
    #       epochs=100,
    #       batch_size=128,
    #       callbacks=[cp_callback]
    #          )

    model.fit(train_padded_sequences, train_data[['english','luganda']].values,
        validation_data=(val_padded_sequences ,val_data[['english','luganda']].values),
        epochs=100,
        batch_size=128,
        # callbacks=[cp_callback]
            )
        
    
    
    
    
if __name__ =='__main__':
    run()
