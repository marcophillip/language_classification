from models import transformer
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os
# sys.path.append('/home/g007markphillip/language_classification/src')
from config import Config
# from src.models import transformer

config=Config()
count=config.count
max_length = config.max_length

new_data = pd.read_csv('../data/processed/processed_1.csv')

train_data=new_data[new_data.folds.isin([0,1,2])].reset_index(drop=True)
# train_data=shuffle(train_data).reset_index().drop(columns='index')
val_data = new_data[new_data.folds==3].reset_index().drop(columns='index')
test_data = new_data[new_data.folds==4].reset_index().drop(columns='index')

tokenizer = Tokenizer(num_words=count)
tokenizer.fit_on_texts(train_data.text)

train_sequences=tokenizer.texts_to_sequences(train_data.text)
train_padded_sequences = pad_sequences(train_sequences,maxlen=max_length,padding='post',truncating='post')

val_sequences=tokenizer.texts_to_sequences(val_data.text)
val_padded_sequences = pad_sequences(val_sequences,maxlen=max_length,padding='post',truncating='post')

test_sequences=tokenizer.texts_to_sequences(test_data.text)
test_padded_sequences = pad_sequences(test_sequences,maxlen=max_length,padding='post',truncating='post')

cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=,
                    save_freq='epoch')

model_names={
    'transformer': transformer()
}
def run(model_name):
    model = model_names[model_name]
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(train_padded_sequences, train_data[['english','acholi','luganda','lumasaba','runyankore','swahili']].values,
          validation_data=(val_padded_sequences ,val_data[['english','acholi','luganda','lumasaba','runyankore','swahili']].values),
          epochs=5,
          batch_size=512,
          callbacks=[cp_callback])
    
    
    
    
    
if __name__ =='__main__':
    run('transformer')