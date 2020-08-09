#######################################################################################
# Yelp - review text preprocessing
#######################################################################################

#########################################
# Dependencies
#########################################

import argparse
from datetime import datetime
import pandas as pd

# For statistics
from collections import Counter

# For preprocessing
import string
from sklearn.feature_extraction import text

####################################
# Functions
####################################

def print_time ():
    # Print date time for performance
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

def statistics (df):

    print("Number of rows in dataframe:", len(df["text"]))
    print ("Min number of words per review:" , df["word count"].min())
    print ("Average number of words per review:" , round(df ["word count"].mean()))
    print ("Max number of words per review:" , df ["word count"].max())
    print ("Total number of words in all reviews:" , df ["word count"].sum())
      # Find distinct number of words
    results = Counter()
    df["text"].str.lower().str.split().apply(results.update)
    print("Number of distinct words in all reviews: ", len(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Preprocess a csv file containing Yelp text reviews'
    )

    parser.add_argument(
        'in_reviews_file',
        type=str,
        help='The input reviews file (stratified)'
    )

    parser.add_argument(
        'out_reviews_file',
        type=str,
        help='The output reviews file (pre-processed)'
    )

    parser.add_argument(
        'word_frequency_threshold',
        type=str,
        help='The frequency below which words will be dropped)'
    )
    args = parser.parse_args()

    ####################################
    # Load file
    ####################################
    print_time ()
    print ("-----------------------------------------------------------------")
    print ("Load reviews file ")
    print ("-----------------------------------------------------------------")

    in_reviews_csv_file = args.in_reviews_file
    out_reviews_csv_file = args.out_reviews_file
    word_frequency_threshold = int(args.word_frequency_threshold)

    print ('in_reviews_csv_file=' ,  in_reviews_csv_file)

    df_reviews = pd.read_csv( in_reviews_csv_file
                             , delimiter = '|'
                             , quotechar = "'"
                             , escapechar = '\\'
                             #, nrows = 100
                             )

    print ("Columns in dataframe: ", list(df_reviews.columns))
    # Add a column with the number of words per review
    df_reviews ["word count"] = df_reviews["text"].str.split().str.len()

    ####################################
    # Initial statistics
    ####################################
    print ("-----------------------------------------------------------------")
    print ("Initial statistics")
    print ("-----------------------------------------------------------------")
    statistics(df_reviews)

    ####################################
    # Preprocessing
    ####################################
    print ("-----------------------------------------------------------------")
    print ("Preprocess the data")
    print ("-----------------------------------------------------------------")
    print_time ()

    print ('1. Remove punctuation...')

    print('Punctuation characters:', string.punctuation)

    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text

    df_reviews["clean_text"] = df_reviews['text'].apply(remove_punctuations)

    print ('2. Convert to lowercase...')
    df_reviews["lower_text"] = df_reviews["clean_text"].str.lower()

    print ('3. Remove stopwords...')

    # Import stopwords with scikit-learn (as it has more stopwords than NLTK)
    stop = text.ENGLISH_STOP_WORDS

    # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
    df_reviews['text_without_stopwords'] = df_reviews['lower_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    #print(df_reviews['text_without_stopwords'])

    print_time ()
    print ('4. Remove rarely used words ...')

    # Create pandas series for word frequency across all reviews
    ps_wc = df_reviews['text_without_stopwords'].str.split(expand=True).stack().value_counts()
    # Convert the word frequency series to a dataframe
    df_reviews_wc = pd.DataFrame(ps_wc, columns=['wcount'])
    df_reviews_wc['name'] = df_reviews_wc.index

    # List of words with count higher than the threshold
    high_count_words = df_reviews_wc [df_reviews_wc["wcount"] > word_frequency_threshold ]["name"].values.tolist()
    print ('Words appearing more times than the threshold (',word_frequency_threshold, '):', len(high_count_words))

    print_time ()
    print("Removing low count words...")
    # Derive a new dataframe column where only the words appearing more than the threshold overall are kept
    df_reviews['text_without_rarewords'] = df_reviews['text_without_stopwords'].apply(lambda x: ' '.join([word for word in x.split() if word in (high_count_words)]))
    print_time ()

    df_reviews ["word count"] = df_reviews["text_without_rarewords"].str.split().str.len()

    print_time ()

    print ('5. Tidy up ...')
    # Rename the text column
    # do it before creating review_df to avoid 'SettingWithCopyWarning'
    df_reviews.rename(columns={'text': 'original_text', 'text_without_rarewords': 'text'}, inplace=True)

    # Drop columns no longer required, rather than create a new dataframe view 'review_df'
    # Setting values in 'review_df' (if it's a view), will result in 'SettingWithCopyWarning', and the results may not be as expected
    df_reviews.drop(df_reviews.columns.difference(['date','review_id','business_id' , 'text', 'stars', 'label', 'word count']), 1,
        inplace=True)

    df_reviews_out = df_reviews

    ####################################
    # Final statistics
    ####################################
    print ("-----------------------------------------------------------------")
    print ("Final statistics after pre-processing")
    print ("-----------------------------------------------------------------")
    statistics(df_reviews_out)

    ####################################
    # Save preprocessed file to csv
    ####################################
    with open( out_reviews_csv_file, 'w') as csv_file:
        df_reviews_out.to_csv(path_or_buf=csv_file)