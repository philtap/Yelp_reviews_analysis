#!/bin/bash

./Execute_Review_Preprocess.sh  /home/hduser/Desktop/DMML2/yelp_dataset/sample_reviews/reviews_stratified.csv /home/hduser/Desktop/DMML2/yelp_dataset/processed_review_data/reviews_preprocessed.csv 5 > Execute_Review_Preprocess.log

./uninstall.ksh >Execute_Review_Preprocess.log 2>&1