#######################################################################################
# Yelp - review text analysis
#######################################################################################

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

# For preprocessing
import string

from pprint import pprint

# For ML models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import text
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



####################################
# 1. Load the data
####################################
# Data corresponds to one review part file (000.part)

print ("-----------------------------------------------------------------")
print ("1. Load the data")
print ("-----------------------------------------------------------------")

# Todo
# 1.  Path and filename to parameterise

####################################
# 1.a Load on part file
####################################

filename = '../../yelp_dataset/reviews.csv/000.part'
df = pd.read_csv(filename)

####################################
# 1.b Load ALL the data using dask
####################################
#import dask.dataframe as dd

# # This is slow , disabling for now
# ddf = dd.read_csv('../yelp_dataset/reviews.csv/*.part')
# print(ddf.shape)
# ddf.head()


########################################
# 2. Investigate
########################################

print ("-----------------------------------------------------------------")
print ("2. Explore the data")
print ("-----------------------------------------------------------------")

print ("Columns in dataframe: ", list(df.columns))
print("Number of rows in dataframe:", len(df["text"]))

# Add a column with the number of words per review
df ["word count"] = df["text"].str.split().str.len()

print('----Statistics about initial reviews----')
print ("Min number of words per review:" , df ["word count"].min())
print ("Average number of words per review:" , round(df ["word count"].mean()))
print ("Max number of words per review:" , df ["word count"].max())
print ("Total number of words in all reviews:" , df ["word count"].sum())

# Find distinct number of words
results = Counter()
df["text"].str.lower().str.split().apply(results.update)
print(" Number of distinct words in all reviews: ", len(results))

###########################################################
# 3. Pre-processing
###########################################################

# More Preprocessing of the review text could be done
# - Positive/Negative words?
#  See Lesson 3 : Sentiment analysis (FromScratch)


print ("-----------------------------------------------------------------")
print ("3. Preprocess the data")
print ("-----------------------------------------------------------------")

print ('3.1. Remove punctuation...')

print('Punctuation characters:', string.punctuation)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df["clean_text"] = df['text'].apply(remove_punctuations)

print ('3.2. Convert to lowercase...')
df["lower_text"] = df["clean_text"].str.lower()

print ('3.3. Remove stopwords...')

# Import stopwords with scikit-learn (as it has more stopwords than NLTK)
stop = text.ENGLISH_STOP_WORDS

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
df['text_without_stopwords'] = df['lower_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print(df['text_without_stopwords'])

print ('3.4. Remove rarely used words ...')

# Create pandas series for word frequency across all reviews
ps_wc = df['text_without_stopwords'].str.split(expand=True).stack().value_counts()
# Convert the word frequency series to a dataframe
df_wc = pd.DataFrame(ps_wc, columns=['wcount'])
df_wc['name'] = df_wc.index

# List of words with count higher than 5
high_count_words = df_wc [df_wc["wcount"] > 5 ]["name"].values.tolist()
print ('Words appearing more than 5 times')
# print(high_count_words)
print('Number of words:', len(high_count_words))

print_time ()
print("Removing low count words...")
# Derive a new dataframe column where only the words appearing more than 5 times overall are kept
df['text_without_rarewords'] = df['text_without_stopwords'].apply(lambda x: ' '.join([word for word in x.split() if word in (high_count_words)]))
print_time ()

print(df['text_without_rarewords'])

df ["word count"] = df["text_without_rarewords"].str.split().str.len()

print ('3.5. Tidy up ...')
# Rename the text column
# do it before creating review_df to avoid 'SettingWithCopyWarning'
df.rename(columns={'text': 'original_text', 'text_without_rarewords': 'text'}, inplace=True)

# Drop columns no longer required, rather than create a new dataframe view 'review_df'
# Setting values in 'review_df' (if it's a view), will result in 'SettingWithCopyWarning', and the results may not be as expected
df.drop(df.columns.difference(['date','review_id','business_id' , 'text', 'stars', 'label', 'word count']), 1,
        inplace=True)

review_df = df

first_lines=review_df.iloc[0:10]["text"]

# At the end,  count the numbers of words left per review and on average.

print('----Statistics about preprocessed reviews----')

review_df ["word count"] = review_df["text"].str.split().str.len()

print ("Min number of words per review:" , review_df ["word count"].min())
print ("Average number of words per review:" , round(review_df ["word count"].mean()))
print ("Max number of words per review:" , review_df ["word count"].max())
print ("Total number of words in all reviews:" , review_df ["word count"].sum())

# Find distinct number of words
results = Counter()
review_df["text"].str.lower().str.split().apply(results.update)
print("Number of distinct words in all reviews: ", len(results))

print (review_df["text"])

# The number of words (min, max, average) has been divided by 2
# More preprocessing is needed

########################################
# 4. Try a base model
########################################

print ("-----------------------------------------------------------------")
print ("4. Try a base model")
print ("-----------------------------------------------------------------")

# The collection of texts is also called a corpus in NLP.
# The vocabulary in this case is a list of words that occurred in our text where each word has its own index.
# The resulting vector will be with the length of the vocabulary and a count for each word in the vocabulary.


print ("-----------------------------------------------------------------")
print ("4.1 Baseline linear regression model on all reviews")
print ("-----------------------------------------------------------------")

# # Defining a Baseline Model
# # so that we can compare with more advanced models
#

# # Restrict to 10000 rows
# #review_df = review_df[0:10000]
#
# reviews = review_df['text'].values
# y = review_df['stars'].values
#
# ####################
#
# # First, split the data into a training and testing set
# reviews_train, reviews_test, y_train, y_test = train_test_split( reviews, y, test_size=0.25, random_state=1000)
#
# # Here we will use the BOW model to vectorize the sentences.
# # You can use the CountVectorizer
# # Since testing data may not be available during training, create the vocabulary using only the training data.
# # Using this vocabulary,  create the feature vectors for each sentence of the training and testing set:
#
# vectorizer = CountVectorizer()
# vectorizer.fit(reviews_train)
#
# X_train = vectorizer.transform(reviews_train)
# X_test  = vectorizer.transform(reviews_test)
#
# # Show the vocabulary
# print(vectorizer.vocabulary_)
#
# print ('Size of the vocabulary:',len(vectorizer.vocabulary_))
#
# # Size of the training vector
# print(X_train.shape)
#
#
# print_time ()
#
# reg= LinearRegression()
# reg.fit(X_train, y_train)
#
# # Make predictions using the testing set
# y_pred = reg.predict(X_test)
#
# # The coefficients
# print('Coefficients: \n', reg.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(y_test, y_pred))
# print_time ()

print ("-----------------------------------------------------------------")
print ("4.2 Baseline logistic regression model on all reviews")
print ("-----------------------------------------------------------------")

# Create a column Label with the binary labels: 0 for 1,2 Stars  and 1 for 3,4,5 Stars
# def binary_classification (x):
#     if float(x) <=2:
#         return 0
#     else:
#         return 1
#
# review_df['label'] = (review_df['stars'].apply(binary_classification))
#
# reviews = review_df['text'].values
# y = review_df['label'].values
#
# ####################
#
# # First, to split the data into a training and testing set
# reviews_train, reviews_test, y_train, y_test = train_test_split( reviews, y, test_size=0.25, random_state=1000)
#
# # Here we will use the BOW model to vectorize the sentences.
# # You can use the CountVectorizer for this task.
# # Since you might not have the testing data available during training, you can create the vocabulary using only the training data.
# # Using this vocabulary, you can create the feature vectors for each sentence of the training and testing set:
#
# vectorizer = CountVectorizer()
# vectorizer.fit(reviews_train)
#
# X_train = vectorizer.transform(reviews_train)
# X_test  = vectorizer.transform(reviews_test)
#
# # Show the vocabulary
# print(vectorizer.vocabulary_)
#
# print ('Size of the vocabulary:',len(vectorizer.vocabulary_))
#
# # Size of the training vector
# print('Size of the training vector:',X_train.shape)
#
# # The classification model we are going to use is the logistic regression
# # as labels are in [0,1]
#
# print_time ()
#
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
#
# print("Accuracy:", score)
#
# print_time ()
#

print ("-----------------------------------------------------------------")
print ("4.3 Baseline Classification model (Random forest) on all reviews")
print ("-----------------------------------------------------------------")

reviews = review_df['text'].values
y = review_df['stars'].values

# First, split the data into a training and testing set
reviews_train, reviews_test, y_train, y_test = train_test_split( reviews, y, test_size=0.25, random_state=1000)

#################################################################################################
# COMMENTED OUT
#

#
# # Use BOW model to vectorize the sentences (CountVectorizer)
# # Since testing data may not be available during training, create the vocabulary using only the training data.
# # Using this vocabulary,  create the feature vectors for each sentence of the training and testing set:
#
# vectorizer = CountVectorizer()
# vectorizer.fit(reviews_train)
#
# X_train = vectorizer.transform(reviews_train)
# X_test  = vectorizer.transform(reviews_test)
#
# # Show the vocabulary
# # print(vectorizer.vocabulary_)
#
# print ('Size of the vocabulary:',len(vectorizer.vocabulary_))
#
# # Size of the training vector
# print(X_train.shape)
#
# print_time ()
#
# classifier = RandomForestClassifier(n_estimators=100)
# classifier.fit(X_train, y_train)
#
# # Make predictions using the testing set
# y_pred = classifier.predict(X_test)
#
# print (y_test)
#
# print (y_pred)
#
# # Create confusion matrix
# tab = pd.crosstab(y_test, y_pred, rownames=['Actual Stars'], colnames=['Predicted Stars'])
# print(tab)
#
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#
#
#
# # Pararameters used by our current forest
# print('Parameters currently in use:\n')
# pprint(classifier.get_params())
#
# print_time ()
# COMMENTED OUT - end
#################################################################################################

print ("-----------------------------------------------------------------")
print ("4.4 Neural networks on all reviews")
print ("-----------------------------------------------------------------")

#################################################################################################
# COMMENTED OUT
# print_time ()
#
# input_dim = X_train.shape[1]  # Number of features
#
# model = Sequential()
# model.add(layers.Dense(5, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])
#
#
# history = model.fit(X_train, y_train,
#                     epochs=10,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
#
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# test_loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# print("Loss:  {:.4f}".format(test_loss))
# model.summary()
#
# print_time ()
# COMMENTED OUT - end
#################################################################################################
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
#
# plot_history(history)

print ("-----------------------------------------------------------------")
print ("4.5 Neural networks (Word embeddings) on all reviews")
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
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

print_time ()


loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

print_time ()

#plot_history(history)




##########################################################
#  Possible next steps
# - Find ways of reducing the dimensions of the NLP review text features
#       Remove numbers
#        Remove words that do not appear often
# - Need to try a NN algorithm
#  See article : https://realpython.com/python-keras-text-classification/
# - then need to look at recurrent NN
# - look into Sentiment analysis
#   See chapter 3 of https://github.com/bmtgoncalves/FromScratch/
##########################################################





