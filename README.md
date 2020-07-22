# Credit-Card-Fraud-Detection

In this project I build machine learning model to identify fraud in **Credit card transactions**. I also make several data visualizations to reveal patterns and structure in the data.

## Introduction

  The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not. Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.
  It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.<br />
  Most of the transactions are not fraudulent which makes it really hard for detecting the fraudulent transactions....
  
## Softwares and Libraries

  * Python
  * numpy
  * pandas
  * matplotlib
  * seaborn
  
## Dataset

  You can download the dataset used in this project from https://www.kaggle.com/mlg-ulb/creditcardfraud
  
## What to predict:

  For each transaction in the test set,Our model will be trained to predict whether or not the transaction is fraudulent or not
  
## Built With
  
  In this Project **Random Forest** Model is used
  
## Inside The Project

  #### Creating Training and Testing Data set

  ```python
  x_train,x_test,y_train,y_test = train_test_split(xData,yData,test_size = 0.2, random_state=42)

  ```
  #### Training the model

  ```python
  rf = RandomForestClassifier()
  rf.fit(x_train,y_train)
  ```
  #### Making the prediction

  ```python
  y_pred = rf.predict(x_test)
  ```

# Accuracy
```python
print("Accuracy Using Random Forest Classsifier ",acc_gn)
```
Accuracy Using Random Forest Classifier **99.95%**
