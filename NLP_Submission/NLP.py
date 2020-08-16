import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'The Main NLP process'
    )

    parser.add_argument(
        'in_csv_file',
        type=str,
        help='The input CSV file delimited by "|"'
    )

    parser.add_argument(
        'in_glove',
        type=str,
        help='The input pre-trained glove embeddings'
    )

    parser.add_argument(
        'Out_Model',
        type=str,
        help='The output model location'
    )

    args = parser.parse_args()

    in_csv_file = args.in_csv_file
    in_glove = args.in_glove
    Out_Model = args.Out_Model

    print('Loading and processing input file')
    In_Reviews = pd.read_csv(in_csv_file
                        ,   delimiter = '|'
                        ,   quotechar = "'"
                        ,   escapechar = '\\'
                        )

    #Encoding the variable to account for ordinal properties
    In_Reviews['stars'] = In_Reviews['stars'].apply(int)
    In_Reviews = pd.get_dummies(In_Reviews , columns = ['stars'])

    #Change index
    In_Reviews.index = In_Reviews['review_id']


    #Apply stemming
    In_Reviews_ABT = In_Reviews[['text','stars_1','stars_2','stars_3','stars_4','stars_5']]


    #Get train and test split
    In_Reviews_ABT['weight'] = 1
    train = In_Reviews_ABT.sample(n=320000, random_state=198666, weights='weight')
    test = In_Reviews_ABT[~In_Reviews_ABT.index.isin(train.index)]


    # I'm using GLoVe word vectors to get pretrained word embeddings
    embed_size = 100
    max_features = 10000
    maxlen = 100

    embedding_file = in_glove

    # read in embeddings
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))

    #
    class_names = ['stars_1','stars_2','stars_3','stars_4','stars_5']
    Y_train = train[class_names].values
    Y_test = test[class_names].values

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train['text'].values))
    X_train = tokenizer.texts_to_sequences(train['text'].values)
    X_test = tokenizer.texts_to_sequences(test['text'].values)
    x_train = pad_sequences(X_train, maxlen = maxlen)
    x_test = pad_sequences(X_test, maxlen = maxlen)


    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    missed = []
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missed.append(word)


    #Lets train the model!!!
    if os.path.exists(Out_Model) == False:
        print('Training Model')
        inp = Input(shape = (maxlen,))
        x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)
        x = SpatialDropout1D(0.5)(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        outp = Dense(5, activation = 'sigmoid')(avg_pool)

        model = Model(inputs = inp, outputs = outp)
        earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0, patience =3)
        checkpoint = ModelCheckpoint(monitor = 'val_loss' , save_best_only = True, filepath=Out_Model)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, Y_train, batch_size = 1024, epochs = 10, validation_split=0.1
              )
        model.save(Out_Model)

    else:
        print('Model already exists, no need to retrain')


    print('Predicting on test set')
    local = tf.keras.models.load_model(Out_Model)
    y_predict = local.predict([x_test], batch_size=1024, verbose =1)

    y_actual_stars = np.argmax(test[['stars_1','stars_2','stars_3','stars_4','stars_5']].values, axis=1) + 1
    y_predict_stars = np.argmax(y_predict, axis=1) + 1

    print('Confusion Matrix')
    print(metrics.confusion_matrix(y_actual_stars, y_predict_stars))

    print('Accuracy Score')
    print(metrics.accuracy_score(y_actual_stars, y_predict_stars))

    print('Classification Report')
    print(metrics.classification_report(y_actual_stars, y_predict_stars))