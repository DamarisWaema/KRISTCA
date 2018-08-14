#scikit learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
#for doing text clustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import adjusted_rand_score
from os import listdir
#pdfminer imports
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import sklearn
import numpy as np
import time
import mysql.connector
from nltk.corpus import stopwords
from sklearn.feature_extraction import text

#items required for clustering
searchString = "text clustering"
dbTablesToCluster = ["Researchers", "ResearchProjects", "UpcomingEvents"]
CVsToClusterFolder = "D:/IRS/KRIS/public/storage/CVs"
researchPapersToClusterFolder = "D:/IRS/KRIS/public/storage/ResearchPapers"
docsToCluster=[]
researchPapersToClusterList=[]
researchersToCluster=[]
fundingOpportunitiesToCluster=[]
FOpportunitiesToCluster=[]
i=1
top10WordsPerCluster=[]
top10WordsPerResearcherCluster=[]
top10WordsPerFundingOpportunityCluster=[]
textDoClusters=[]
researchersClusters=[]
fundingOpportunitiesClusters=[]
relevantDocClusterIndices=[]
relevantResearchersClusterIndices=[]
relevantFundingOpportunitiesClusterIndices=[]
rankedRelevantResearchPapersToReturn=[]
rankedRelevantResearchersToReturn=[]
rankedRelevantFundingOpportunitiesToReturn=[]
#function to get list of files from a folder
def list_files(directory, extension):
        return (f for f in listdir (directory) if f.endswith ('.' + extension))
def removeStopWords(sentence):
    #stop_words = text.ENGLISH_STOP_WORDS.union (my_additional_stop_words)
    stop_words = text.ENGLISH_STOP_WORDS
    searchstringSet=set([])
    for word in sentence.split():
        searchstringSet.add(word)
    searchstringSetWithoutStopWords=searchstringSet-stop_words
    returnstring=''
    for word in searchstringSetWithoutStopWords:
        if(returnstring==''):
            returnstring = returnstring + word
        else:
            returnstring =returnstring+' '+ word
    return returnstring
def rankRelevantResearchPapers():
    print ("BEGINNING OF RANKING RESEARCH PAPERS")
    docsToRank = []
    searchStringWithoutStopWords = removeStopWords (searchString)
    indicesForRelevantDocs = []
    for clusterIndex in relevantDocClusterIndices:
        cluster = np.where (textDoClusters == clusterIndex)  # don't forget import numpy as np
        cluster0Docindices = cluster[0]

        for index in cluster0Docindices:
            docsToRank.append (docsToCluster[index])
            indicesForRelevantDocs.append (index)

    x = len (docsToRank)
    docsToRank.append (searchStringWithoutStopWords)
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    docsToRank_tfidf_matrix = tfidf_vectorizer.fit_transform (docsToRank)
    similarityMeasure = cosine_similarity (docsToRank_tfidf_matrix[x], docsToRank_tfidf_matrix).tolist ()
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    relevantDocsIndices = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantDocsIndices.append (indicesForRelevantDocs[smi])
        smi = smi + 1
    "Cluster indices for relevant clusters"
    print (relevantDocsIndices)

    sortedRelevantDocsIndices = sorted (relevantDocsIndices, reverse=True)
    print (sortedRelevantDocsIndices)
    for index in sortedRelevantDocsIndices:
        rankedRelevantResearchPapersToReturn.append (researchPapersToClusterList[index])



    print ('Done Ranking  Relevant Research Papers')

def rankRelevantResearchers():
    print('BEGINNING OF RANKING AND DISPLAYING RELEVANT RESEARCHERS')
    researchersToRank = []
    searchStringWithoutStopWords = removeStopWords (searchString)
    indicesForRelevantResearchers = []
    for clusterIndex in relevantResearchersClusterIndices:
        cluster = np.where (researchersClusters == clusterIndex)  # don't forget import numpy as np
        clusterNResearchersindices = cluster[0]

        for index in clusterNResearchersindices:
            researchersToRank.append (researchersToCluster[index])
            indicesForRelevantResearchers.append (index)

    x = len (researchersToRank)
    researchersToRank.append (searchStringWithoutStopWords)
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    researchersToRank_tfidf_matrix = tfidf_vectorizer.fit_transform (researchersToRank)
    similarityMeasure = cosine_similarity (researchersToRank_tfidf_matrix[x], researchersToRank_tfidf_matrix).tolist ()
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    relevantResearchersIndices = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantResearchersIndices.append (indicesForRelevantResearchers[smi])
        smi = smi + 1
    "Cluster indices for relevant clusters"
    print (relevantResearchersIndices)

    sortedRelevantResearchersIndices = sorted (relevantResearchersIndices, reverse=True)
    print (sortedRelevantResearchersIndices)
    for index in sortedRelevantResearchersIndices:
        rankedRelevantResearchersToReturn.append (researchersToCluster[index])


    print ('Done Ranking Relevant Researchers')

def rankRelevantFundingOpportunities():
    print ('BEGINNING OF RANKING RELEVANT FUNDING OPPORTUNITIES')
    fundingOpportunitiesToRank = []
    searchStringWithoutStopWords = removeStopWords (searchString)
    indicesForRelevantFundingOpportunities = []
    for clusterIndex in relevantFundingOpportunitiesClusterIndices:
        cluster = np.where (fundingOpportunitiesClusters == clusterIndex)  # don't forget import numpy as np
        clusterNFundingOpportunityindices = cluster[0]

        for index in clusterNFundingOpportunityindices:
            fundingOpportunitiesToRank.append (fundingOpportunitiesToCluster[index])
            indicesForRelevantFundingOpportunities.append (index)

    x = len (fundingOpportunitiesToRank)
    fundingOpportunitiesToRank.append (searchStringWithoutStopWords)
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    fundingOpportunitiesToRank_tfidf_matrix = tfidf_vectorizer.fit_transform (fundingOpportunitiesToRank)
    similarityMeasure = cosine_similarity (fundingOpportunitiesToRank_tfidf_matrix[x], fundingOpportunitiesToRank_tfidf_matrix).tolist ()
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    relevantFundingOpportunityIndices = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantFundingOpportunityIndices.append (indicesForRelevantFundingOpportunities[smi])
        smi = smi + 1
    "Cluster indices for relevant clusters"
    print (relevantFundingOpportunityIndices)

    sortedRelevantResearchersIndices = sorted (relevantFundingOpportunityIndices, reverse=True)
    print (sortedRelevantResearchersIndices)
    for index in sortedRelevantResearchersIndices:
        rankedRelevantFundingOpportunitiesToReturn.append (fundingOpportunitiesToCluster[index])

    print ('Done Ranking Relevant Funding Opportunities')

def rankingAlgorithm():
    print("Ranking Portion of The Algorithm")
    rankRelevantResearchPapers()
    rankRelevantResearchers()
    rankRelevantFundingOpportunities()


def multiLevelTextLusteringAlgorithm():
    print("THIS ALGORITHM SHOULD CLUSTER BOTH SQL DATA AND TEXT DOCUMENTS")
    sqlClusteringAlgorithmUsingKMeans()
    #textDocCluteringUsingKMeansThatPrintsTopWordsPerCluster ()
    textDocClusteringUsingKMeans()
    #textdocClusteringUsingMiniBatchKMeans()

def clusterResearchers():
    print()
    print("RESULTS OF CLUSTERING RESEARCHERS")
    numberOfClusters = 6
    connection = mysql.connector.connect (host='127.0.0.1', user='root', password='', database='TESTKRIS')
    cursor = connection.cursor ()
    query = (
        "SELECT name, DptName, RIName, ResearchAreaOfInterest, AboutResearcher FROM Researchers INNER JOIN users on Researchers.User_ID = users.id INNER JOIN Departments ON Researchers.DepartmentID =Departments.Department_ID INNER JOIN researchInstitutions ON Researchers.ResearchInstitutionID =researchInstitutions.ResearchInstitution_ID ")
    cursor.execute (query)
    researchers = cursor.fetchall ()
    for row in researchers:
        researchersToCluster.append (str (row))
        # researcher=str(row)
        # print(researcher)
    vectorizer = TfidfVectorizer (stop_words='english')
    X = vectorizer.fit_transform (researchersToCluster)

    print ("RESULTS OF CLUSTERING RESEARCHERS USING K-MEANS ALGORITHM")
    km = sklearn.cluster.KMeans (init='k-means++', max_iter=500, n_init=1,
                                 verbose=0, n_clusters=numberOfClusters)
    t1 = time.time ()
    clusters = km.fit_predict (X)
    clusteringTime = time.time () - t1
    global researchersClusters
    researchersClusters=clusters
    print ("TOTAL TIME TAKEN TO CLUSTER RESEARCHERS using kmeans: " + str (t1))
    # Note that your input data has dimensionality m x n and the clusters array has dimensionality m x 1 and contains the indices for every document
    # print (X.shape)
    # print (clusters.shape)
    # Example to get all documents in cluster 0
    y = 0
    while y < numberOfClusters:
        cluster_0 = np.where (clusters == y)  # don't forget import numpy as np
        cluster0indices = cluster_0[0]
        print ("Researchers in cluster " + str (y))

        for clusterindex in cluster0indices:
            print (researchersToCluster[clusterindex])
        y = y + 1
        print (" ")
    # cluster_0 now contains all indices of the documents in this cluster, to get the actual documents you'd do:
    X_cluster_0 = X[cluster_0]
    # print(X_cluster_0)
    # getting top 10 words per researcher cluster
    print ("TOP TERMS PER CLUSTER:")
    order_centroids = km.cluster_centers_.argsort ()[:, ::-1]
    terms = vectorizer.get_feature_names ()
    for i in range (5):
        topWords = ""
        #print ("Cluster %d:" % i, )
        x = 1
        for ind in order_centroids[i, :10]:
            if (topWords == ""):
                topWords = topWords + terms[ind]

            else:
                topWords = topWords + " " + terms[ind]

            #print (' %s' % terms[ind], )
            #print ()

            if (x == 10):
                top10WordsPerResearcherCluster.append (topWords)
                x = 1
                print (top10WordsPerResearcherCluster)
            x = x + 1
    print ("Done getting top ten words in researcher clusters as documents")

def clusterFundingOpportunities():
    print ()

    print ("RESULTS OF CLUSTERING FUNDING OPPORTUNITIES")
    numberOfClusters = 4
    connection = mysql.connector.connect (host='127.0.0.1', user='root', password='', database='TESTKRIS')
    cursor = connection.cursor ()
    query = (
        "SELECT FunderName, ResearchAreasFunded FROM FundingOpportunities INNER JOIN Funders on FundingOpportunities.Funder_ID = Funders.Funder_ID ")
    cursor.execute (query)
    fundingOpportunities = cursor.fetchall ()
    for row in fundingOpportunities:
        fundingOpportunitiesToCluster.append (str (row))
        # researcher=str(row)
        # print(researcher)
    vectorizer = TfidfVectorizer (stop_words='english')
    X = vectorizer.fit_transform (fundingOpportunitiesToCluster)

    print ("RESULTS OF CLUSTERING FUNDING OPPORTUNITIES USING K-MEANS ALGORITHM")
    km = sklearn.cluster.KMeans (init='k-means++', max_iter=500, n_init=1,
                                 verbose=0, n_clusters=numberOfClusters)
    t1 = time.time ()
    clusters = km.fit_predict (X)
    clusteringTime = time.time () - t1
    global fundingOpportunitiesClusters
    fundingOpportunitiesClusters = clusters
    print ("TOTAL TIME TAKEN TO CLUSTER FUNDING OPPORTUNITIES using kmeans: " + str (t1))
    # Note that your input data has dimensionality m x n and the clusters array has dimensionality m x 1 and contains the indices for every document
    # print (X.shape)
    # print (clusters.shape)
    # Example to get all documents in cluster 0
    y = 0
    while y < numberOfClusters:
        cluster_0 = np.where (clusters == y)  # don't forget import numpy as np
        cluster0indices = cluster_0[0]
        print ("Funding Opportunities in cluster " + str (y))

        for clusterindex in cluster0indices:
            print (fundingOpportunitiesToCluster[clusterindex])
        y = y + 1
        print ()
    # cluster_0 now contains all indices of the documents in this cluster, to get the actual documents you'd do:
    X_cluster_0 = X[cluster_0]
    # print(X_cluster_0)
    # getting top 10 words per researcher cluster
    print ("TOP TERMS PER CLUSTER:")
    order_centroids = km.cluster_centers_.argsort ()[:, ::-1]
    terms = vectorizer.get_feature_names ()
    for i in range (4):
        topWords = ""
        print ("Cluster %d:" % i, )
        x = 1

        for ind in order_centroids[i, :4]:
            if (topWords == ""):
                topWords = topWords + terms[ind]

            else:
                topWords = topWords + " " + terms[ind]

            # print (' %s' % terms[ind], )
            # print ()

            if (x == 4):
                top10WordsPerFundingOpportunityCluster.append (topWords)
                x = 1
                print (top10WordsPerFundingOpportunityCluster)
            x = x + 1
    print ("Done getting top ten words in Funding Opportunities clusters as documents")

def sqlClusteringAlgorithmUsingKMeans():
    print("RESULTS OF CLUSTERING SQL DATA")
    clusterResearchers()
    clusterFundingOpportunities()


def textDocCluteringUsingKMeansThatPrintsTopWordsPerCluster():
    print("TEXT CLUSTERING USING KMEANS ALGORITHM THAT PRINTS TOP WORDS PER CLUSTER")
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

def matchingResearchPaperClusters():
    print('BEGINNING OF MATCHING RESEARCH PAPER CLUSTERS')
    searchStringWithoutStopWords = removeStopWords (searchString)
    print ("Matching Using Cosine Similarity Measure")
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    x = len (top10WordsPerCluster)
    top10WordsPerCluster.append (searchStringWithoutStopWords)
    clusters_tfidf_matrix = tfidf_vectorizer.fit_transform (top10WordsPerCluster)
    #print (clusters_tfidf_matrix.shape)
    similarityMeasure = []
    similarityMeasure = cosine_similarity (clusters_tfidf_matrix[x], clusters_tfidf_matrix).tolist ()
    print (top10WordsPerCluster)
    print ("Similarity measure when compared with the user's search string")
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    sortedSimilarityMeasureAsSingleList = sorted (similarityMeasure[0], reverse=True)
    print (SimilarityMeasureAsSingleList)
    print (sortedSimilarityMeasureAsSingleList)
    relevantClusterIndicesForDocs = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantClusterIndicesForDocs.append (smi)
        smi = smi + 1
    "Cluster indices for relevant clusters"
    #print (relevantClusterIndicesForDocs)
    global relevantDocClusterIndices
    relevantDocClusterIndices = sorted (relevantClusterIndicesForDocs, reverse=True)
    print("Relevant document cluster indices ")
    print (relevantDocClusterIndices)
    print ("Done Matching Research papers")

def matchingResearcherClusters():
    print()
    print('BEGINNING OF MATCHING RESEARCHERS')
    searchStringWithoutStopWords = removeStopWords (searchString)
    print ("Matching Using Cosine Similarity Measure")
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    x = len (top10WordsPerResearcherCluster)
    top10WordsPerResearcherCluster.append (searchStringWithoutStopWords)
    clusters_tfidf_matrix = tfidf_vectorizer.fit_transform (top10WordsPerResearcherCluster)
    #print (clusters_tfidf_matrix.shape)
    similarityMeasure = []
    similarityMeasure = cosine_similarity (clusters_tfidf_matrix[x], clusters_tfidf_matrix).tolist ()
    print (top10WordsPerResearcherCluster)
    print ("Similarity measure when compared with the user's search string")
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    sortedSimilarityMeasureAsSingleList = sorted (similarityMeasure[0], reverse=True)
    print (SimilarityMeasureAsSingleList)
    print (sortedSimilarityMeasureAsSingleList)
    relevantResearcherClusterIndices = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantResearcherClusterIndices.append (smi)
        smi = smi + 1
    "Cluster indices for relevant clusters"
    print("Relevant Researcher Cluster Indices in any order")
    print (relevantResearcherClusterIndices)
    global relevantResearchersClusterIndices
    relevantResearchersClusterIndices = sorted (relevantResearcherClusterIndices, reverse=True)
    print("Relevant Researcher Cluster Indices in order of relevance starting with most relevant")

    print (relevantResearchersClusterIndices)
    print ("Done Matching Researchers")

def matchingFundingOpportunitiesCluster ():
    print ()
    print ('BEGINNING OF MATCHING Funding Opportunities')
    searchStringWithoutStopWords = removeStopWords (searchString)
    print ("Matching Using Cosine Similarity Measure")
    tfidf_vectorizer = TfidfVectorizer (stop_words='english')
    x = len (top10WordsPerFundingOpportunityCluster)
    top10WordsPerFundingOpportunityCluster.append (searchStringWithoutStopWords)
    clusters_tfidf_matrix = tfidf_vectorizer.fit_transform (top10WordsPerFundingOpportunityCluster)
    # print (clusters_tfidf_matrix.shape)
    similarityMeasure = []
    similarityMeasure = cosine_similarity (clusters_tfidf_matrix[x], clusters_tfidf_matrix).tolist ()
    print (top10WordsPerFundingOpportunityCluster)
    print ("Similarity measure when compared with the user's search string")
    SimilarityMeasureAsSingleList = similarityMeasure[0]
    SimilarityMeasureAsSingleList.pop ()
    sortedSimilarityMeasureAsSingleList = sorted (similarityMeasure[0], reverse=True)
    print (SimilarityMeasureAsSingleList)
    print (sortedSimilarityMeasureAsSingleList)
    relevantFundingOpportunityClusterIndices = []
    smi = 0
    while (smi < len (SimilarityMeasureAsSingleList)):
        if SimilarityMeasureAsSingleList[smi] > 0:
            relevantFundingOpportunityClusterIndices.append (smi)
        smi = smi + 1
    "Cluster indices for relevant clusters"
    print ("Relevant Researcher Cluster Indices in any order")
    print (relevantFundingOpportunityClusterIndices)
    global relevantFundingOpportunitiesClusterIndices
    relevantFundingOpportunitiesClusterIndices = sorted (relevantFundingOpportunityClusterIndices, reverse=True)
    print ("Relevant Funding Opportunities Cluster Indices in order of relevance starting with most relevant")

    print (relevantFundingOpportunitiesClusterIndices)
    print ("Done Matching Funding Opportunities")

def matchingAlgorithm():
    print('BEGINNING OF MATCHING')
    matchingResearchPaperClusters ()
    matchingResearcherClusters()
    matchingFundingOpportunitiesCluster()

def displaySearchResults():
    print()
    print ("SEARCH RESULTS")

    print("Search string: {}".format(searchString))

    if (len(rankedRelevantResearchersToReturn)==0):
        print('There are no researchers in the database that match your search query')
        print()
    else:
        print ("Relevant Researchers In Order of relevance")
        for researcher in rankedRelevantResearchersToReturn:
            print (researcher)
        print()

    if(len(rankedRelevantResearchPapersToReturn)==0):
        print('There are no research papers in the database that match your search query')
        print('')

    else:
        print ("Relevant Research Papers In Order of relevance")
        for paper in rankedRelevantResearchPapersToReturn:
            print (paper)
        print('')
    if(len(rankedRelevantFundingOpportunitiesToReturn)==0):
        print("There are no Funding Opportunities in the databases that match your search query")
    else:
        print("Relevant Funding Opportunities according to your search string")
        for fundingOpportunity in rankedRelevantFundingOpportunitiesToReturn:
            print (fundingOpportunity)
def textDocClusteringUsingKMeans():
    print()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(docsToCluster)

    print("RESULTS OF CLUSTERING DOCUMENTS USING K-MEANS ALGORITHM")
    km = sklearn.cluster.KMeans (init='k-means++', max_iter=500, n_init=2,
                                 verbose=0, n_clusters=5)
    t1=time.time()
    clusters = km.fit_predict (X)
    global textDoClusters
    textDoClusters=clusters
    clusteringTime=time.time()-t1
    print("TOTAL TIME TAKEN TO CLUSTER DOCUMENTS using kmeans: "+str(t1))
    # Note that your input data has dimensionality m x n and the clusters array has dimensionality m x 1 and contains the indices for every document
    #print (X.shape)
    #print (clusters.shape)
    # Example to get all documents in cluster 0
    y = 0
    while y < 5:
        cluster_0 = np.where (clusters ==y)  # don't forget import numpy as np
        cluster0indices=cluster_0 [0]
        print("Documents in cluster "+ str(y))

        for clusterindex in cluster0indices:
            print(researchPapersToClusterList[clusterindex])
        y=y+1
        print(" ")
    # cluster_0 now contains all indices of the documents in this cluster, to get the actual documents you'd do:
    X_cluster_0 = X[cluster_0]
   # print(X_cluster_0)

    #Printing top 20 words per cluster
    print ("TOP TERMS PER CLUSTER:")
    order_centroids = km.cluster_centers_.argsort ()[:, ::-1]
    terms = vectorizer.get_feature_names ()
    for i in range (5):
        topWords=""
        print ("Cluster %d:" % i, )
        x = 1
        for ind in order_centroids[i, :10]:
            if(topWords==""):
                topWords = topWords +terms[ind]

            else:
                topWords = topWords + " " + terms[ind]

            #print (' %s' % terms[ind], )
            #print ()

            if (x == 10):
                top10WordsPerCluster.append (topWords)
                x=1
                print(top10WordsPerCluster)
            x=x+1
    print("Done getting top ten words as documents")
def textdocClusteringUsingMiniBatchKMeans():
    print("RESULTS OF CLUSTERING USING MINIBATCH KMEANS ALGORITHM")
    batch_size=45
    vectorizer = TfidfVectorizer (stop_words='english')
    X = vectorizer.fit_transform (docsToCluster)
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=5, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)

    t1 = time.time ()
    clusters=mbk.fit_predict(X)
    clusteringTime = time.time () - t1
    print ("TOTAL TIME TAKEN TO CLUSTER DOCUMENTS using minibatch kmeans: " + str (t1))
    # Note that your input data has dimensionality m x n and the clusters array has dimensionality m x 1 and contains the indices for every document
    print (X.shape)
    print (clusters.shape)
    # Example to get all documents in cluster 0
    y = 0
    while y < 5:
        cluster_0 = np.where (clusters == y)  # don't forget import numpy as np
        cluster0indices = cluster_0[0]
        print ("Documents in cluster " + str (y))

        for clusterindex in cluster0indices:
            print (researchPapersToClusterList[clusterindex])
        y = y + 1
        print (" ")
    # cluster_0 now contains all indices of the documents in this cluster, to get the actual documents you'd do:
    X_cluster_0 = X[cluster_0]
    #print (X_cluster_0)


def mainfunction():
    extractTextFromPDFResearchPapers()
    multiLevelTextLusteringAlgorithm()

    matchingAlgorithm()
    rankingAlgorithm()
    displaySearchResults()

class PdfConverter:

    def __init__(self, file_path):
        self.file_path = file_path

    # convert pdf file to a string which has space among words
    def convert_pdf_to_txt(self):
        rsrcmgr = PDFResourceManager ()
        retstr = StringIO ()
        codec = 'utf-8'  # 'utf16','utf-8'
        laparams = LAParams ()
        device = TextConverter (rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open (self.file_path, 'rb')
        interpreter = PDFPageInterpreter (rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set ()
        for page in PDFPage.get_pages (fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                       check_extractable=True):
            interpreter.process_page (page)
        fp.close ()
        device.close ()
        str = retstr.getvalue ()
        retstr.close ()
        return str

    # convert pdf file text to string and save as a text_pdf.txt file
    def save_convert_pdf_to_txt(self):
        content = self.convert_pdf_to_txt ()
        txt_pdf = open ('text_pdf.txt', 'wb')
        txt_pdf.write (content.encode ('utf-8'))
        txt_pdf.close ()

def extractTextFromPDFResearchPapers():
    if __name__ == '__main__':
        researchPapersToCluster = list_files (researchPapersToClusterFolder, "pdf")
        print ("Research Paper to be clustered")
        k=1
        for rp in researchPapersToCluster:
            if(k<=25):
                #print (i, rp)
                #print ("Research Paper text extracted")
                pdfConverter = PdfConverter (file_path=researchPapersToClusterFolder+'/'+rp)
                filecontent=pdfConverter.convert_pdf_to_txt ()
                 # print (filecontent)
                docsToCluster.append(filecontent)
                researchPapersToClusterList.append(rp)

                k=k+1

mainfunction()
