#!/bin/bash

echo **********************************************************
echo      Script:Execute_Review_Preprocess.sh
echo **********************************************************

echo ----------------------------------------------------------
echo            Input parameters
echo ----------------------------------------------------------
echo Input Review file path and name: $1
echo Output review subset file path and name: $2
echo Word Frequency Threshold: $3

echo ----------------------------------------------------------
echo            Preparation
echo ----------------------------------------------------------

echo ----------------------------------------------------------
echo Running Review_Preprocess.py ....
echo ----------------------------------------------------------
#wget https://raw.githubusercontent.com/philtap/....Review_Preprocess.py
python3 Review_Preprocess.py $1 $2 $3
# rm Review_Preprocess.py

echo ----------------------------------------------------------
echo Cleanup
echo ----------------------------------------------------------
