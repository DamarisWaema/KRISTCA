items=[0.2, 0.4, 0.1]
sortedItems=sorted(items)
print (sortedItems)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
university=''
documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"The sun in the sky sky sky is sky sky sky sky sky bright",

"We can see the shining sun, the bright sun"
)
def getSimilarityMeasure():
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print (tfidf_matrix.shape)

    similarityMeasure=cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).tolist()
    similarityMeasureAsList=similarityMeasure[0]


    print(similarityMeasure)
    print(type(similarityMeasure))
    print(type(similarityMeasureAsList))
    print(similarityMeasureAsList)
    similarityMeasureAsList.pop()
    print(similarityMeasureAsList)

    print(sorted(similarityMeasureAsList, reverse=True))

def univerisityName():
    global university
    university='JKUAT'
#getSimilarityMeasure()
univerisityName()
print (university)
print("univeristy name has been printed")
#print("University name: ")+university
print("university name: {}".format(university))