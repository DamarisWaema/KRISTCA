from sklearn.feature_extraction.text import TfidfVectorizer
#for doing text clustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from os import listdir
import PyPDF2
#import textract
import collections


#

searchString = "Text Clustering"
dbTablesToCluster = ["Researchers","ResearchProjects","UpcomingEvents"]
CVsToClusterFolder = "D:/IRS/KRIS/public/storage/CVs"
researchPapersToClusterFolder= "D:/IRS/KRIS/public/storage/ResearchPapers"
docsToCluster=[]
i=1

def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def multiLevelTextLusteringAlgorithm():
    print("Clustering")
#clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(docsToCluster)
#cluster documents using KM algorithm
    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

#print top terms per cluster clusters
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print ("Cluster %d:" % i,)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind],)
            print()

def matchingAlgorithm():
    print("Matching")

def displaySearchResults():
    print ("Dispaying")

#A function that calls all the other functions

def mainfunction():

    multiLevelTextLusteringAlgorithm()
    matchingAlgorithm()
    displaySearchResults()


researchPapersToCluster = list_files(researchPapersToClusterFolder, "pdf")
print("Research Papers to be clustered")

for rp in researchPapersToCluster:
    print(i, rp)

    pdfFileObj = open(researchPapersToClusterFolder+'/'+rp, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    number_of_pages = pdfReader.getNumPages()
    c = collections.Counter(range(number_of_pages))
    print(pdfReader.numPages)
    # creating a page object
    pageObj = pdfReader.getPage(0)
    print("Extracting text .............")
    # extracting text from page
    print(pageObj.extractText())

    # closing the pdf file object
    pdfFileObj.close()
   # for i in number_of_pages:
    #  page = pdfReader.getPage(i)
    #  page_content = page.extractText()
    #  print( page_content.encode('utf-8'))
    #file = open(researchPapersToClusterFolder+'/'+rp,'r', encoding="UTF-8")
    #filecontent=file.read()
    #docsToCluster.append(filecontent)
    #file.close()
   # print (file.read())
    i=1+i

#mainfunction()


