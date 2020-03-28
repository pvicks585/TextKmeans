import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def convert_lower_case(document):
    '''
    Convert each document to lowercase characters
    '''
    document = np.char.lower(document)
    return document


def remove_apostrophe(document):
    '''
    Remove the apostrophe from every document
    '''
    document = np.char.replace(document, "'", "")
    return document

def remove_punctuation(document):
    '''
    Remove every punctuation symbol from every document
    '''
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        document = np.char.replace(document, i, '')
    return document


def number_to_words(document):
    '''
    Convert numbers to word form in each docuemnt
    '''
    tokens = word_tokenize(document)
    for word in tokens:
        if word.isnumeric():
            num_word = num2words(word)
            document = document.replace(word, num_word)
    return document


def remove_stop_words(document):
    '''
    Remove stop words from each document
    '''
    stop_words = stopwords.words('english')
    new_text = ''
    tokens = word_tokenize(document)
    for word in tokens:
        if word not in stop_words:
            new_text = new_text + " " + word
    return new_text
  
              
def remove_single_characters(document):
    '''
    Remove characters of length 1 from each document
    '''
    new_text = ""
    tokens = word_tokenize(document)
    for word in tokens:
        if len(word) > 1:
            new_text = new_text + " " + word
    return new_text



def preprocess(df, column):
    '''
    Run the formulas above for a select column in a dataframe
    '''
    df[column] = df[column].apply(lambda doc: convert_lower_case(doc))
    print('Words have been lowercased')
    
    df[column] = df[column].apply(lambda doc: remove_apostrophe(doc))
    print('Apostrophe Removed')
    
    df[column] = df[column].apply(lambda doc:remove_punctuation(doc))
    print('Punctuation Removed')
    
    df[column] = df[column].apply(lambda doc:number_to_words(doc))
    print('Numbers converted to Words')
    
    df[column] = df[column].apply(lambda doc:remove_stop_words(doc))
    print('Stopwords Removed')
    
    df[column] = df[column].apply(lambda doc:remove_single_characters(doc))
    print('Single Characters Removed')
    
    return df


df = pd.read_csv('/Users/paulvicks/Documents/Articles.csv', encoding= 'unicode_escape')
processed_df = preprocess(df, 'Heading')



def tfidf(df, header):
    '''
    Call the vectorizer
    Fit transform the chosen column in the dataframe
    Call interesting statistics
    '''
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[header])

    idf = vectorizer.idf_
    
    names_weights_dict = (dict(zip(vectorizer.get_feature_names(), idf)))
    frame = pd.DataFrame(names_weights_dict.items())
    
    return vectorizer, tfidf_matrix, vectorizer.vocabulary_, tfidf_matrix.toarray(), frame






def kmeans_optimizer(matrix, K=range(1,40)):
    Sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(matrix)
        Sum_of_squared_distances.append(km.inertia_)
        print('{} is done'.format(k))
        
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()



def kmeans_func(df, matrix, vectorizer, k=30):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=30,
                verbose=1)
    km.fit(matrix)

    df['Cluster'] = km.labels_
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(k):
        print("Cluster {}:".format(i))
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind]))
    return df






