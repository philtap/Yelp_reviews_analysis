echo -**********************************************************-
echo      Script:Execute_NLP.sh
echo -**********************************************************-
##################################################################################################################################################
#./Execute_NLP.sh reviews_stratified.csv  ~/Desktop/DMML2/yelp_dataset/glove_word_embeddings/glove.6B.50d.txt > ./logs/Execute_NLP_2020_08_18_09_50.log
###################################################################################################################################################
echo -**********************************************************-
echo            Input parameters
echo -**********************************************************-

echo Input stratified sample review file - path and name: $1
echo Pre-trained GloVe text embeddings - path and name: $2

echo -**********************************************************-
echo Running Text_Preprocessing.py....
echo -**********************************************************-
echo "Executing text pre-processing"
python3 Text_Preprocessing.py $1 Reviews_Preprocessed.csv

echo -**********************************************************-
echo Random Forest - Base model ....
echo Executing Review_Preprocess_FrequentWords.py...
echo -**********************************************************-

echo "Executing pre-processing (remove frequent words)"
python3 Review_Preprocess_FrequentWords.py Reviews_Preprocessed.csv Reviews_frequent_words.csv 5

echo -**********************************************************-
echo Random Forest - Base model ....
echo Executing Review_Analysis.py...
echo -**********************************************************-

echo "Running Random Forest model (model number 1) to get a baseline"
python3 Review_Analysis.py Reviews_frequent_words.csv 1 0

echo -**********************************************************-
echo Running NLP.py - preprocessed reviews....
echo -**********************************************************-
echo "Executing NLP using the preprocessed file"
python3 NLP.py Reviews_Preprocessed.csv $2 Model_Preprocessed.hdf5

echo -**********************************************************-
echo Running NLP.py - raw reviews ....
echo -**********************************************************-
echo "Executing NLP using the Raw file"
#python3 NLP.py $1 $2 Model_Raw.hdf5


