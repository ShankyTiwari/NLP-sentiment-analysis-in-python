# Importing the libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# To count the iterations 
from tqdm import tqdm

# Importing the dataset
dataset = pd.read_csv('Reviews.csv')

# Dropping the dups in dataset
dataset = dataset.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

def removeHTMLTags(review):
    soup = BeautifulSoup(review, 'lxml')
    return soup.get_text()

def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase

def removeAlphaNumericWords(review):
     return re.sub("\S*\d\S*", "", review).strip()
 
def removeSpecialChars(review):
     return re.sub('[^a-zA-Z]', ' ', review)

def scorePartition(x):
    if x < 3:
        return 0
    return 1

def doTextCleaning(review):
    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review) 
    # Lower casing
    review = review.lower()  
    #Tokenization
    review = review.split()
    #Removing Stopwords and Lemmatization
    lmtzr = WordNetLemmatizer()
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)    
    return review

# Generalizing the score
actualScore = dataset['Score']
positiveNegative = actualScore.map(scorePartition) 
dataset['Score'] = positiveNegative

# creating the document corpus
corpus = []   
for index, row in tqdm(dataset.iterrows()):
    review = doTextCleaning(row['Text'])
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

#Creating a tranform
cv = CountVectorizer(ngram_range=(1,3), max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predict the sentiment for new review
def predictNewReview():
    newReview = input("Type the Review: ")
    
    if newReview =='':
        print('Invalid Review')  
    else:
        newReview = doTextCleaning(newReview)
        new_review = cv.transform([newReview]).toarray()  
        prediction =  classifier.predict(new_review)
        print(prediction)
        if prediction[0] == 1:
            print( "Positive Review" )
        else:        
            print( "Negative Review")

