#!/bin/bash
########################################################################################################################
#
# Script: Execute_Review_Analysis.sh
#
# This is the shell script to use to execute the Review_Analysis.py program
#
########################################################################################################################
# Parameters:
# 1. Input Review local file path and name: $1
# 2. Input Model number to run: $2
# 3. Number of epochs to run: $3
#
# See full list of models in Review_Analysis.py
#
# Example of usage: Run CNN model with embedding  5
# ./Execute_Review_Analysis.sh /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed.csv 5 5
#
########################################################################################################################

echo **********************************************************
echo      Script:Execute_Review_Analysis.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review local file path and name: $1
echo Model number to run: $2
echo Number of epochs to run: $3
echo ----------------------------------------------------------
echo Running Review_Analysis.py ....
echo ----------------------------------------------------------
python3 Review_Analysis.py $1 $2 $3


