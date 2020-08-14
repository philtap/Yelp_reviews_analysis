#######################################################################################
# Yelp - review text analysis
#######################################################################################
####################################
# Requirements
####################################

import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


from pprint import pprint

# For ML models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


####################################
# Functions
####################################

def print_time ():
    # Print date time for performance
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

def plot_history (history):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run ML models on preprocessed Yelp text reviews'
    )

    parser.add_argument(
        'in_reviews_file',
        type=str,
        help='The input reviews file (preprocessed)'
    )
    parser.add_argument(
        'in_model_number',
        type=str,
        help='The model number to run'
    )

    parser.add_argument(
        'in_number_epochs',
        type=str,
        help='Number of epochs to run'
    )

    args = parser.parse_args()
    in_reviews_csv_file = args.in_reviews_file
    model_to_run = int(args.in_model_number)
    number_epochs = int(args.in_number_epochs)

    print ('Review file: ' ,  in_reviews_csv_file)
    print ('Model to run: ' ,  model_to_run)
    print ('Number of epochs (for NN): ' ,  in_reviews_csv_file)

    ####################################
    # Load the data
    ####################################

    print ("-----------------------------------------------------------------")
    print ("Load the data")
    print ("-----------------------------------------------------------------")

    print ('in_reviews_csv_file=' ,  in_reviews_csv_file)
    df = pd.read_csv(in_reviews_csv_file)

    print('Remove any review with no words')
    print("Number of rows in dataframe:", len(df["text"]))
    #print(df[df['text'].isnull()])
    df=df.dropna()
    print("Number of rows in dataframe:", len(df["text"]))
    #print(df[df['text'].isnull()])

    ########################################
    # Try a base model
    ########################################

    reviews = df['text'].values
    y = df['stars'].values

    # Split the data into a training and testing set
    reviews_train, reviews_test, y_train, y_test = train_test_split( reviews, y, test_size=0.25, random_state=1000)

    # Use BOW model to vectorize the sentences (CountVectorizer)
    # Since testing data may not be available during training, create the vocabulary using only the training data.
    # Using this vocabulary,  create the feature vectors for each sentence of the training and testing set:

    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_train)

    X_train = vectorizer.transform(reviews_train)
    X_test  = vectorizer.transform(reviews_test)

    # Show the vocabulary
    # print(vectorizer.vocabulary_)

    print ('Size of the vocabulary:',len(vectorizer.vocabulary_))

    print_time ()

    if model_to_run == 1:

        # The collection of texts is also called a corpus in NLP.
        # The vocabulary in this case is a list of words that occurred in our text where each word has its own index.
        # The resulting vector will be with the length of the vocabulary and a count for each word in the vocabulary.

        print ("-----------------------------------------------------------------")
        print ("Baseline Classification model (Random forest) on all reviews")
        print ("-----------------------------------------------------------------")

        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = classifier.predict(X_test)

        # Create confusion matrix
        tab = pd.crosstab(y_test, y_pred, rownames=['Actual Stars'], colnames=['Predicted Stars'])
        print(tab)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        # Pararameters used by our current forest
        print('Parameters currently in use:\n')
        pprint(classifier.get_params())

        print_time ()

    if model_to_run == 2:

        print ("-----------------------------------------------------------------")
        print ("Simple Neural network model")
        print ("-----------------------------------------------------------------")

        print_time ()

        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(5, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])


        history = model.fit(X_train, y_train,
                            epochs=number_epochs,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        test_loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        print("Loss:  {:.4f}".format(test_loss))
        model.summary()

        print_time ()

    if model_to_run == 3:

        print ("-----------------------------------------------------------------")
        print ("Neural networks (Word embeddings) ")
        print ("-----------------------------------------------------------------")

        print_time ()

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(reviews_train)

        X_train = tokenizer.texts_to_sequences(reviews_train)
        X_test = tokenizer.texts_to_sequences(reviews_test)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

        print(reviews_train[2])
        print(X_train[2])

        # for word in ['the', 'all', 'happy', 'sad']:
        #      print(word,':',tokenizer.word_index[word])



        # Use 500 as the CURRENT Max number of words per review: 448
        maxlen = 500

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        print(X_train[0, :])

        embedding_dim = 50

        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   input_length=maxlen))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        print_time ()
        history = model.fit(X_train, y_train,
                            epochs=number_epochs,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)

        print_time ()


        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        plot_history (history)

        print_time ()

    if model_to_run == 4:
        print('Model 4')

    if model_to_run == 5:
        print('Model 5')

    if model_to_run == 6:
        print('Model 6')









