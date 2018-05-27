# Sentence-Classification

This project aims at sentence classification for the UCI dataset (https://archive.ics.uci.edu/ml/datasets/Sentence+Classification).
Naive Bayes model is used for the given problem because of varrious advantages it has over traditional classification algorithms like SVM.

The rationale behind using this model is :
  * it relies on a very simple representation of the document (called the bag of words representation)
  * the bias–variance tradeoff. Spam/sentiment/text classification type data are often noisy and usually high-dimensional (more predictors than samples, n≪p). The naive assumption that predictors are independent of one another is a strong, high-bias, one. NB is observed to give better performance in such situations.
  
max_features is the hyperparameter which is tweaked to get the better result:
  * max_features=1500 gives 71.15% accuracy
  * max_features=2000 gives 76.44% accuracy
  * max_features=3000 gives 80.13% accuracy
  * max_features=5000 gives 80.77% accuracy
Therefore, max_features=3000 is selected. Bigger values enhaced the performance very little but required considerable bigger memory. So, they were discarded.

Therefore, accuracy of 80.13% is achieved with the Naive Bayes model.

Note: Better accuracy can be achieved with Deep Learning models like RNN with LSTM. That is out of my scope right now :D
