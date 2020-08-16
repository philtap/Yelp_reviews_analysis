#!/bin/bash

# This file should be run from the NLP directory in the submission zip
echo -**********************************************************-
echo      Script:Install_NLP.sh
echo -**********************************************************-

echo ------------------------------------------
echo Installing dependencies....
echo ------------------------------------------
pip3 install -r requirements.txt

echo ------------------------------------------
echo Downloading NLTK stopwords
echo ------------------------------------------
python3 -m nltk.downloader stopwords

echo ------------------------------------------------
echo Downloading pre-trined word embeddings from glove
echo ------------------------------------------------
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir glove
mv glove.6B.100d.txt ./glove
mv glove.6B.50d.txt ./glove
rm glove.6B*
