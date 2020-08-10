#!/bin/bash
########################################################################################################################
#
# Script: Execute_Review_Analysis.sh
#
########################################################################################################################
# Parameters:
# 1. Input Review local file path and name: $1
#
# Example of usage:
# ./Execute_Review_Analysis.sh /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed.csv
#
########################################################################################################################

echo **********************************************************
echo      Script:Execute_Review_Analysis.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review local file path and name: $1

echo ----------------------------------------------------------
echo Running Review_Analysis.py ....
echo ----------------------------------------------------------
#wget https://raw.githubusercontent.com/philtap/Yelp_reviews_analysis/master/Review_Analysis/Review_Analysis.py
python3 Review_Analysis.py $1
#rm Review_Analysis.py

