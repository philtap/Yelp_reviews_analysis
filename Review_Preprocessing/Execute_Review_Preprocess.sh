#!/bin/bash
########################################################################################################################
#
# Script: Execute_Review_Preprocess.sh
#
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

echo **********************************************************
echo      Script:Execute_Review_Preprocess.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review local file path and name: $1
echo Output review local subset file path and name: $2
echo Word Frequency Threshold: $3

echo ----------------------------------------------------------
echo Running Review_Preprocess.py ....
echo ----------------------------------------------------------
wget https://raw.githubusercontent.com/philtap/Yelp_reviews_analysis/master/Review_Preprocessing/Review_Preprocess.py
python3 Review_Preprocess.py $1 $2 $3
rm Review_Preprocess.py

