#!/bin/bash
########################################################################################################################
#
# Script: Execute_Review_Preprocess.sh
#
# This is the shell script to use to execute the Review_Preprocess.py program
# See description in the .py file
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
python3 Review_Preprocess.py $1 $2 $3


