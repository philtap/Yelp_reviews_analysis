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
# 2. Location and name of pretrained text embeddings 50d (Glove): $2
# 3. Input Model number to run: $3
# 4. Number of epochs to run: $4
#
# See full list of models in Review_Analysis.py
#
# Example of usage: Run CNN model (5) with embedding and 5 epochs
# ./Execute_Review_Analysis.sh ./processed_review_data/reviews_preprocessed.csv ./glove/glove.6B.50d.txt 5 5
########################################################################################################################

echo **********************************************************
echo      Script:Execute_Review_Analysis.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review local file - path and name: $1
echo Input pretrained word embeddings file  - path and name: $2
echo Model number to run: $3
echo Number of epochs to run: $4
echo ----------------------------------------------------------
echo Running Review_Analysis.py ....
echo ----------------------------------------------------------
python3 Review_Analysis.py $1 $2 $3 $4


