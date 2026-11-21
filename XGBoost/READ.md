# Credit Card Fraud Detection

The project is a model running on public data, done to showcase
model can predict whenever the transaction was a fraud

## About it

Model is XGBoost, boosting model that sums the weak learners
boosting its prediction. The data contains 31 columns, 3 of which
are Time, Amount and Class.Rest is private information hidden
inside PCA.The data had no missing values. The dataset is highly
unbalanced, the frauds accounts for 0,172% of all transactions. I used
train_test_split method on it. I used probability cutoff in order
to get my results.


## MODEL EVALUATION

--Accuracy:  99.94908886626172

--Precision is:  94.8051948051948

--Recall is:  74.48979591836735

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Here is the link to the dataset i was working on



