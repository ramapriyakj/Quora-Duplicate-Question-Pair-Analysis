import pickle
import re

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import config as c

nltk.download('stopwords')
nltk.download('wordnet')
stops = set(stopwords.words('english')).union(gensim.parsing.preprocessing.STOPWORDS)
pd.options.display.max_colwidth = 1000

quora_duplicate_questions = c.quora_duplicate_questions
quora_pre_processed_file = c.c_quora_pre_processed_file
quora_bag_of_words_file = c.c_quora_bag_of_words_file
quora_tf_idf_file = c.c_quora_tf_idf_file
quora_lda_bow_file = c.c_quora_lda_bow_file
quora_lda_tfidf_file = c.c_quora_lda_tfidf_file
max_features = c.c_max_features


def handle_pickle(pickle_path, operation="read", pickle_data=None):
    """
    Method to store and retrieve the pickle files
    :param pickle_path: Pickle file path
    :param operation: Read or Write
    :param pickle_data: Data to store in pickle file
    :return: pickle data
    """
    if operation == "read":
        try:
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("Exception in loading pickle file {} due to the following exception {}".format(pickle_path, e))
            return None
    else:
        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(pickle_data, f)
        except Exception as e:
            print("Exception in dumping pickle file {} due to the following exception {}".format(pickle_path, e))
            return None


def preprocess(text):
    """
    Method to pre process the string.
    :param text: Original text
    :return: text
    """
    # lower text
    text = str(text).lower()

    # unidecode text
    text = unidecode.unidecode(text)

    # handle contractions
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"ain\'t", "is not", text)

    # handle symbols and currencies
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    text = text.replace(",000,000", "m").replace(",000", "k").replace("%", " percent ").replace("₹", " rupee ").replace(
        "$", " dollar ").replace("€", " euro ")

    # remove symbols
    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    # remove whitespace
    text = re.sub(r"[\s]+", " ", text)

    # Tokenize text
    text = text.split()

    # Remove stop words, lemmatize, only keep words with length > 4
    text_list = []
    for w in text:
        if w not in stops:
            w_ = WordNetLemmatizer().lemmatize(w, pos='v')
            if len(w_) > 4:
                text_list.append(w_)

    text = ' '.join(text_list)

    # Return np.nan if length of text is 0 or total tokens are less than 3
    if len(text) == 0 or len(text.split()) < 3:
        return np.nan

    return text


def load_preprocessed_data(source_frame, pre_processed_file, overwrite=False):
    """
    Method to load the preprocessed text file.
    :param source_frame: The original data frame to apply preprocessing
    :param pre_processed_file: Location to store the pre processed file
    :param overwrite: If set to True, source_frame will be processed. Else data will be retrieved from
                      pre_processed_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: preprocess_data
    """
    try:
        preprocess_data = None
        if not overwrite:
            preprocess_data = handle_pickle(pre_processed_file)

        if preprocess_data is None:
            preprocess_data = source_frame.copy()
            preprocess_data['processed'] = preprocess_data['question'].apply(lambda x: preprocess(x))
            preprocess_data = preprocess_data.dropna()
            handle_pickle(pre_processed_file, "write", preprocess_data)
        return preprocess_data

    except Exception as e:
        raise Exception("Unable to do pre processing due to exception {}".format(e))


def load_features(feature_type, feature_file, source_data, max_features=250, overwrite=False):
    """
    Method to generate features
    :param feature_type: Feature type to generate - '.bow' or '.tfidf'
    :param feature_file: Location to store the feature file
    :param source_data: preprocessed data frame
    :param max_features: Maximum features to generate
    :param overwrite: If set to True, source_data will be processed. Else data will be retrieved from
                      feature_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: features, feature_vectorizer
    """
    try:
        features, feature_vectorizer = None, None
        if not overwrite:
            features, feature_vectorizer = handle_pickle(feature_file)

        if features is None:
            if feature_type == '.bow':
                feature_vectorizer = CountVectorizer(min_df=5, max_df=0.8, max_features=max_features)

            elif feature_type == '.tfidf':
                feature_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, max_features=max_features)

            feature_vectorizer.fit(source_data['processed'])
            features = feature_vectorizer.transform(source_data['processed'].values)
            handle_pickle(feature_file, "write", (features, feature_vectorizer))

        return features, feature_vectorizer
    except Exception as e:
        raise Exception("Unable to generate {} features due to exception {}".format(feature_type, e))


def get_features_and_predictions(feature_file, features, topics=20, n_clusters=5, overwrite=False):
    """
    Method applies LatentDirichletAllocation on features and generate LDA features. It also clusters them using kmeans
    and reduce dimensionality of the LDA features to 2 using PCA (useful for projection).
    :param feature_file:  Location to store the LDA features, Cluster labels and PCA features
    :param features: bag of words or tfidf features
    :param topics: Number of topics to generate using LDA
    :param n_clusters: Number of clusters to apply on LDA features
    :param overwrite: If set to True, features will be processed. Else data will be retrieved from
                      feature_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: lda_features, topics, lda_model, kmeans_model, pca_features
    """
    try:
        lda_features = ftopics = lda_model = kmeans_model = pca_features = None
        if not overwrite:
            lda_features, ftopics, lda_model, kmeans_model, pca_features = handle_pickle(feature_file)

        if lda_features is None or ftopics != topics:
            scaler = StandardScaler()
            lda_model = LatentDirichletAllocation(n_components=topics, learning_method='online', random_state=42,
                                                  n_jobs=-1)
            lda_features = lda_model.fit_transform(features)
            lda_features = scaler.fit_transform(lda_features)
            kmeans_model = KMeans(n_clusters=n_clusters)
            pca = PCA(n_components=2)
            kmeans_model.fit(lda_features)
            pca_features = pca.fit_transform(lda_features)
            handle_pickle(feature_file, "write", (lda_features, topics, lda_model, kmeans_model, pca_features))
        return lda_features, topics, lda_model, kmeans_model, pca_features
    except Exception as e:
        raise Exception("Unable to generate lda features and clusters due to exception {}".format(e))


def print_five_questions_for_each_cluster(kmeans_clusters, data, feature_type):
    """
    Method prints 5 questions associated with each cluster
    :param kmeans_clusters: K means model
    :param data: data frame
    :param feature_type: Type of features uon which k means algorithm was fit on
    """
    print("\nFive questions associated with each cluster using {} features".format(feature_type))
    clusters = np.unique(kmeans_clusters)
    for clus in clusters:
        temp_data = data.iloc[np.where(kmeans_clusters == clus)]
        print("\nCluster {} (total records = {})".format(clus, temp_data.shape[0]))
        print("-------------------------------------------------------------------------------")
        print(temp_data['question'].tail().to_string(index=False))
        print("-------------------------------------------------------------------------------")


def print_topics(lda_model, feature_vectorizer, n_top_words, feature_type):
    """
    Method to print top n words associated with each topic
    :param lda_model: LatentDirichletAllocation model
    :param feature_vectorizer: CountVectorizer or TfidfVectorizer
    :param n_top_words: Total words to print
    :param feature_type: Type of feature bog of words or TFIDF (just used for printing)
    """
    print("\nTop {} words associated with topics generated using {} features".format(n_top_words, feature_type))
    words = feature_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda_model.components_):
        print("Topic # {} - {}".format(topic_idx, " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])))


def train(lda_topics=20, topic_clusters=5, overwrite=False):
    """
    Method to generate topics from LatentDirichletAllocation model, Run K means from to perfrom clustering these LDA
    features and apply PCA to visualize these clusters
    :param lda_topics: Number of topics to generate using LDA
    :param topic_clusters: Number of clusters to apply on LDA features
    :param overwrite: If set to True, preprocess files, feature files and LDA feature files will be re processed.
                      Else data will be retrieved from existing files if exist. Need to set it to True when running
                      for first time else will throw file not found exception.
    """
    # Data source
    data_source = pd.read_csv(quora_duplicate_questions, sep='\t').dropna()

    # Concatenate questions
    data = pd.concat((data_source['question1'], data_source['question2'])).unique()
    for i in range(5):
        np.random.seed(35)
        np.random.shuffle(data)
    data = pd.DataFrame(data, columns=['question'])

    # Sample 80% of data randomly
    data = data.sample(frac=0.8, random_state=42)
    del data_source

    # Preprocess data
    preprocess_data = load_preprocessed_data(data, quora_pre_processed_file, overwrite)
    del data
    print("\nData set size : ", preprocess_data.shape)
    print("\nSample data : ")
    print(preprocess_data['question'].head().to_string(index=False))
    print("\nSample preprocessed data : ")
    print(preprocess_data['processed'].head().to_string(index=False))

    # Generate topics, clusters and visualize them using bag of words and TFIDF features
    for model_name in ['Bag_of_words_lda', 'tf_idf_lda']:
        if model_name == 'Bag_of_words_lda':
            features, feature_vectorizer = load_features('.bow', quora_bag_of_words_file, preprocess_data, max_features,
                                                         overwrite)
            lda_features, _, lda_model, kmeans_model, pca_features = get_features_and_predictions(quora_lda_bow_file,
                                                                                                  features, lda_topics,
                                                                                                  topic_clusters,
                                                                                                  overwrite)
        elif model_name == 'tf_idf_lda':
            features, feature_vectorizer = load_features('.tfidf', quora_tf_idf_file, preprocess_data, max_features,
                                                         overwrite)
            lda_features, _, lda_model, kmeans_model, pca_features = get_features_and_predictions(quora_lda_tfidf_file,
                                                                                                  features, lda_topics,
                                                                                                  topic_clusters,
                                                                                                  overwrite)

        clus_labels = kmeans_model.labels_

        print_topics(lda_model, feature_vectorizer, 10, model_name)
        print_five_questions_for_each_cluster(kmeans_model.labels_, preprocess_data, model_name)

        cmap = plt.cm.get_cmap('jet')
        fig, ax = plt.subplots(figsize=(16, 10))
        for i in range(topic_clusters):
            _ = ax.scatter(x=pca_features[np.where(clus_labels == i), 0], y=pca_features[np.where(clus_labels == i), 1],
                           c=np.array(cmap(i / topic_clusters)).reshape(1, -1), label=i)
        ax.legend()
        plt.title(model_name + " - PCA(n_components = 2)")
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.show()


if __name__ == '__main__':
    # Train model and print topics and clusters
    train(lda_topics=20, topic_clusters=5, overwrite=True)
