import pandas as pd
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from replace import replace_functions
from replace_function import replace_all
from abbreviations import replace_abbreviations
from stars import aster
df = pd.read_csv(r'C:\Users\Acer\Desktop\Capstone\FINAL EVALUATION\FINAL PROJECT\capstone_proj\libs\train.csv')
X=df.iloc[:,1].values
y=df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
training=preprocess(X_train)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=0,n_estimators=650)
clf.fit(training,y_train)
from sklearn.pipeline import Pipeline
text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',RandomForestClassifier()),])
text_clf=text_clf.fit(X_train,y_train)
ans=text_clf.predict(X_test)
d=replace_functions()
e=replace_abbreviations()
def listToString(s):  
    str1 = " "    
    return (str1.join(s))   
while(1):
    question =input("Enter your choice as\n 1 for entering sentence\n 2 for exiting:\n ")
    if question=='1':
        sample =str(input("Enter the sentence: "))
        sample=replace_all(sample,e)
        sample=sample.lower()
        sample=replace_all(sample, d)
        sample=sample.split(" ")
        samples=aster(sample)
        sample=listToString(samples)
        print("The message after converting: ", sample)
        review = re.sub('[^a-zA-Z]', ' ', sample)
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        sample=review
        sample =np.array(sample)
        sizes=sample.shape[0]
        #print(sizes)
        text=text_clf.predict(sample)
        #print(text)
        count=0
        list2=[]
        for i in range(0,sizes):
            if text[i]==1:
                list2.append(sample[i])
                count+=1
        prob=count/sizes
        def intersection(lst1, lst2):
            lst3=[]
            for x in lst1:
                if x in lst2:
                    lst3.append('******')
                else:
                    lst3.append(x)
            return lst3
        if prob < 0.2:
            #print("Message may not be hatespeech")
            print("Your message passed is:")
            res = intersection(samples,list2)
            sample2=" ".join(res)
            print(sample2)
        else:
            print("Message is hatespeech")
    else:
        exit(0)
