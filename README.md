# Yelp_reviews_analysis
DMML2 - Yelp Review analysis

## NLP Preprocessing of review text

The Preprocessing script will take a csv file with a (stratified) sample of reviews and apply NLP pr-processing techniques.
Words are converted to lowercase, punctuation and stop words are removed.
Finally words with an overall frequency lower than the input threshold are removed from the review
It outputs a new review csv file, ready for modelling.

The code is in the "Review_Preprocessing" folder:
- Execute_Review_Preprocess.sh 
- Review_Preprocess.py

The sh script should be run from the command line
Parameters:
1. Input review file: local file path (including file name) of the sample reviews csv file
2. Output review file: local file path (including file name) of the output reviews csv file
3. Threshold: the frequency of a word (e.g. 5)  below which a word will be dropped from the data set 
```
Execute_Review_Preprocess.sh <input review file > <output review file> <threshold>
```
## Analysis of review text 

Perform Machine learning analysis of Yelp Restaurant reviews: predict number of Stars (1-5) based on the review text 

The code is in the "Review_Analysis" folder:
- Execute_Review_Analysis.sh 
- Review_Analyis.py

1. Input review file: local file path (including file name) of the sample preprocessed reviews csv file
2. Number of the Machine learning  model to run (1-5)
3. Number of epochs (for Neural Networks models: 2-5)

```
Execute_Review_Analysis.sh <input review file > <model to run> <number of epochs>
```
## Installation
Install dependencies via
```
    pip3 install -r requirements.txt
```



