########################################################################################################################
# Yelp - review text preprocessing
#
#  Review_Preprocess_FrequentWords.py
#
# Origin: https://github.com/philtap/Yelp_reviews_analysis/blob/master/For_submission/Review_Preprocess_FrequentWords.py
#
# This python program is a subset of Review_Preprocess.py
# It takes as input a csv file containing reviews and applies preprocessing to the review text
# using Natural Language Processing (NLP)
# It does the following:
#  - removes words that are not frequent in the vocabulary (set of words found in all reviews)
#   this operation is done using the threshold input parameter. Any words appearing less time than the
#   threshold in the vocabulary is removed form all reviews
#  - removes any review become empty as the result of the above
# The output is placed in the output file (path and name) provided as input parameter 2
########################################################################################################################

########################################################################################################################
# Parameters:
# 1. Input Review local file path and name
# 2. Output review local subset file path and name
# 3. Word Frequency Threshold:
#
# Example of usage:
# python3 Review_Analysis.py  /home/hduser/Desktop/DMML2/yelp_dataset/sample_reviews/reviews_stratified.csv
#                             /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed.csv
#                             5
########################################################################################################################

#########################################
# Dependencies
#########################################

import argparse
from datetime import datetime
import pandas as pd

# For statistics
from collections import Counter

# For preprocessing

####################################
# Functions
####################################

def print_time ():
    # Print date time for performance
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

def statistics (df):
    # This function provide statistics about reviews/words in the data set
    # - total number of reviews,
    # - total number of words
    # - min-avg-max words per review
    # - distinct words in the data set reviews
    # Run before and after the preprocessing, it is useful to evaluate its effect on the data set

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
                             #, nrows = 10000
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
    print ('Remove rarely used words ...')

    # Create pandas series for word frequency across all reviews
    ps_wc = df_reviews['text'].str.split(expand=True).stack().value_counts()
    # Convert the word frequency series to a dataframe
    df_reviews_wc = pd.DataFrame(ps_wc, columns=['wcount'])
    df_reviews_wc['name'] = df_reviews_wc.index

    # List of words with count higher than the threshold
    high_count_words = df_reviews_wc [df_reviews_wc["wcount"] > word_frequency_threshold ]["name"].values.tolist()
    print ('Words appearing more times than the threshold (',word_frequency_threshold, '):', len(high_count_words))

    print("Removing low count words...")
    # Derive a new dataframe column where only the words appearing more than the threshold overall are kept
    df_reviews['text_without_rarewords'] = df_reviews['text'].apply(lambda x: ' '.join([word for word in x.split() if word in (high_count_words)]))
    print_time ()

    df_reviews ["word count"] = df_reviews["text_without_rarewords"].str.split().str.len()

    print ('Tidy up ...')
    print("Rename columns")

    df_reviews.rename(columns={'text': 'original_text', 'text_without_rarewords': 'text'}, inplace=True)

    print("Drop unnecessary columns")
    df_reviews.drop(df_reviews.columns.difference(['date','review_id','business_id' , 'text', 'stars', 'label', 'word count']), 1,
        inplace=True)

    # Remove reviews with empty text
    print("Remove reviews with empty text")
    df_reviews=df_reviews.dropna()

    ####################################
    # Final statistics
    ####################################
    print ("-----------------------------------------------------------------")
    print ("Final statistics after pre-processing")
    print ("-----------------------------------------------------------------")
    statistics(df_reviews)

    ####################################
    # Save preprocessed file to csv
    ####################################
    with open( out_reviews_csv_file, 'w') as csv_file:
        df_reviews.to_csv(path_or_buf=csv_file)