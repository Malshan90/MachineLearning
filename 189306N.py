import pandas as pd
import nltk 
import numpy as np
import string
import nltk
import re 

#Reading the trainig data set
data = pd.read_csv('F:\\MSC\\ML\\train.csv', sep='\t')
len(data)
#Loading the test data set
test = pd.read_csv('F:\\MSC\\ML\\test.csv')

len(test)
#appending the datasets for preprocessing 
appended = data.append(test, ignore_index=True)

#function to remove a given pattern in a string
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    

#Removing mentions in tweets
appended['tidy_tweet'] = np.vectorize(remove_pattern)(appended['Tweet text'], "@[\w]*")

#Removing special characters, numbers, punctuations
appended['tidy_tweet'] = appended['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

#creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(appended['tidy_tweet'])

#Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_feature = bow[:3817,:]
test_feature = bow[3817:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_feature, data['Label'], random_state=42, test_size=0.3)

pred = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)

prediction = pred.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.5
prediction_int = prediction_int.astype(np.int)
# Getting the predictions

test_pred = lreg.predict_proba(test_feature)
test_pred_int = test_pred[:,1] >= 0.5
test_pred_int = test_pred_int.astype(np.int)
test['Label'] = test_pred_int

test.head()

submit = test[['Tweet index','Label']]
submit.to_csv('result.csv', index=False)

