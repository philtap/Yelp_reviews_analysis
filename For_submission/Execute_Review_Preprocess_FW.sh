#!/bin/bash
########################################################################################################################
#
# Script: Execute_Review_Preprocess.sh
#
# This is the shell script to use to execute the Review_Preprocess.py program
# See description in the .py file
#
# Origin: https://github.com/philtap/Yelp_reviews_analysis/blob/master/For_submission/Review_Preprocess_FrequentWords.py
########################################################################################################################
# Parameters:
# 1. Input Review local file path and name: $1
# 2. Output review local subset file path and name: $2
# 3. Word Frequency Threshold: $3
#
# Example of usage:
# ./Execute_Review_Preprocess.sh /home/hduser/Desktop/DMML2/yelp_dataset/sample_reviews/reviews_stratified.csv
#                                /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed.csv
#                                5
########################################################################################################################
#./Execute_Review_Preprocess_FW.sh /home/hduser/Desktop/DMML2/yelp_dataset/sample_reviews/reviews_stratified.csv /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed_fw.csv 5 >
# Execute_Review_Preprocess_FW_5.log

echo **********************************************************
echo      Script:Execute_Review_Preprocess_FW.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review local file path and name: $1
echo Output review local subset file path and name: $2
echo Word Frequency Threshold: $3

echo ----------------------------------------------------------
echo Review_Preprocess_FrequentWords ....
echo ----------------------------------------------------------
python3 Review_Preprocess_FrequentWords.py $1 $2 $3


