import pickle
import re

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import unidecode
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout, BatchNormalization

import config as c

nltk.download('stopwords')
nltk.download('wordnet')
stops = set(stopwords.words('english')).union(gensim.parsing.preprocessing.STOPWORDS)

project_folder = c.project_folder
models_folder = c.d_models_folder
quora_duplicate_questions = c.quora_duplicate_questions
quora_pre_processed_file = c.d_quora_pre_processed_file
quora_bag_of_words_file = c.d_quora_bag_of_words_file
quora_tf_idf_file = c.d_quora_bag_of_words_file
quora_text_to_seq_file = c.d_quora_text_to_seq_file
glove_embedding_text = c.d_glove_embedding_text
max_features = c.d_max_features
rf_max_features = c.d_rf_max_features


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

    # Tokenize
    text = text.split()

    # Remove stop words, lemmatize, only keep words with length > 2
    text_list = []
    for w in text:
        if w not in stops:
            w_ = WordNetLemmatizer().lemmatize(w, pos='v')
            if len(w_) > 2:
                text_list.append(w_)

    text = ' '.join(text_list)

    # Return np.nan if length of text is 0
    if len(text) == 0:
        return np.nan

    return text


def load_preprocessed_data(mode, source_frame, pre_processed_file, overwrite=False):
    """
    Method to load the preprocessed text file.
    :param mode: Preprocessing will be performed on the source_frame.
                 If mode is 'train', pre processed data will be stored under pre_processed_file.
                 else this parameter is ignored. Should be set to None to preprocess test dataframe
    :param source_frame: The original data frame to apply preprocessing
    :param pre_processed_file: Location to store the pre processed file
    :param overwrite: If set to True, source_frame will be processed. Else data will be retrieved from
                      pre_processed_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: preprocess_data[['question1', 'question2']], preprocess_data[['is_duplicate']]
    """
    try:
        preprocess_data = None
        if mode == 'train':
            if pre_processed_file.endswith('.pre'):
                if not overwrite:
                    preprocess_data = handle_pickle(pre_processed_file)
            else:
                raise Exception("Invalid Preprocessed file {}".format(pre_processed_file))

        if preprocess_data is None:
            preprocess_data = source_frame.copy()
            preprocess_data['question1'] = preprocess_data['question1'].apply(lambda x: preprocess(x))
            preprocess_data['question2'] = preprocess_data['question2'].apply(lambda x: preprocess(x))
            preprocess_data = preprocess_data.dropna()

            if mode == 'train':
                handle_pickle(pre_processed_file, "write", preprocess_data)

        return preprocess_data[['question1', 'question2']], preprocess_data[['is_duplicate']]

    except Exception as e:
        raise Exception("Unable to do pre processing due to exception {}".format(e))


def load_bag_of_words_features(mode, source_data, bag_of_words_file, max_features=250, overwrite=False):
    """
    Method to generate bag of words features
    :param mode: Bag of words features are generated from source_data.
                 If mode is 'train', the data will be stored under bag_of_words_file.
                 else this parameter is ignored. Should be set to None to generate bag of words for test dataframe
    :param source_data: preprocessed data frame
    :param bag_of_words_file: Location to store the feature file
    :param max_features: Maximum features to generate
    :param overwrite: If set to True, source_data will be processed. Else data will be retrieved from
                      bag_of_words_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: bow_q1, bow_q2, bow_labels, count_vectorizer
    """
    try:
        if bag_of_words_file.endswith('.bow'):
            bow_q1 = bow_q2 = bow_labels = count_vectorizer = None
            if mode == 'train':
                if not overwrite:
                    bow_q1, bow_q2, bow_labels, count_vectorizer = handle_pickle(bag_of_words_file)
            else:
                _, _, _, count_vectorizer = handle_pickle(bag_of_words_file)
        else:
            raise Exception("Invalid bag of words file {}".format(bag_of_words_file))

        if bow_q1 is None:
            x, y = source_data

            if mode == 'train':
                count_vectorizer = CountVectorizer(max_features=max_features)
                count_vectorizer.fit(pd.concat((x['question1'], x['question2'])).unique())

            bow_q1 = count_vectorizer.transform(x['question1'].values).toarray()
            bow_q2 = count_vectorizer.transform(x['question2'].values).toarray()
            bow_labels = np.asarray(y)

            if mode == 'train':
                handle_pickle(bag_of_words_file, "write", (bow_q1, bow_q2, bow_labels, count_vectorizer))

        return bow_q1, bow_q2, bow_labels, count_vectorizer

    except Exception as e:
        raise Exception("Unable to generate bag of words features due to exception {}".format(e))


def load_tfidf_features(mode, source_data, tfidf_file, max_features=250, overwrite=False):
    """
    Method to generate tfidf features
    :param mode: tfidf features are generated from source_data.
                 If mode is 'train', the data will be stored under tfidf_file.
                 else this parameter is ignored. Should be set to None to generate tfidf features for test dataframe
    :param source_data: preprocessed data frame
    :param tfidf_file: Location to store the feature file
    :param max_features: Maximum features to generate
    :param overwrite: If set to True, source_data will be processed. Else data will be retrieved from
                      tfidf_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: tfidf_q1, tfidf_q2, tfidf_labels, tfidf_vectorizer
    """
    try:
        if tfidf_file.endswith('.tfidf'):
            tfidf_q1 = tfidf_q2 = tfidf_labels = tfidf_vectorizer = None
            if mode == 'train':
                if not overwrite:
                    tfidf_q1, tfidf_q2, tfidf_labels, tfidf_vectorizer = handle_pickle(tfidf_file)
            else:
                _, _, _, tfidf_vectorizer = handle_pickle(tfidf_file)
        else:
            raise Exception("Invalid tfidf file {}".format(tfidf_file))

        if tfidf_q1 is None:
            x, y = source_data

            if mode == 'train':
                tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
                tfidf_vectorizer.fit(pd.concat((x['question1'], x['question2'])).unique())

            tfidf_q1 = tfidf_vectorizer.transform(x['question1'].values).toarray()
            tfidf_q2 = tfidf_vectorizer.transform(x['question2'].values).toarray()
            tfidf_labels = np.asarray(y)

            if mode == 'train':
                handle_pickle(tfidf_file, "write", (tfidf_q1, tfidf_q2, tfidf_labels, tfidf_vectorizer))

        return tfidf_q1, tfidf_q2, tfidf_labels, tfidf_vectorizer

    except Exception as e:
        raise Exception("Unable to generate tfidf features due to exception {}".format(e))


def load_text_to_sequences(mode, source_data, text_to_seq_file, max_features=250, overwrite=False):
    """
    Method to generate sequences for text data using keras tokenizer
    :param mode: sequences are generated from source_data.
                 If mode is 'train', the data will be stored under text_to_seq_file.
                 else this parameter is ignored. Should be set to None to generate sequences for test dataframe
    :param source_data: preprocessed data frame
    :param text_to_seq_file: Location to store the feature file
    :param max_features: Maximum features to generate
    :param overwrite: If set to True, source_data will be processed. Else data will be retrieved from
                      text_to_seq_file location. Need to set it to True when running for first time
                      else will throw file not found exception.
    :return: q1_sequences, q2_sequences, labels, keras_tokenizer
    """
    try:
        if text_to_seq_file.endswith('.seq'):
            q1_sequences = q2_sequences = labels = keras_tokenizer = None
            if mode == 'train':
                if not overwrite:
                    q1_sequences, q2_sequences, labels, keras_tokenizer = handle_pickle(text_to_seq_file)
            else:
                _, _, _, keras_tokenizer = handle_pickle(text_to_seq_file)
        else:
            raise Exception("Invalid text to sequence file {}".format(text_to_seq_file))

        if q1_sequences is None:
            x, y = source_data

            if mode == 'train':
                keras_tokenizer = Tokenizer()
                keras_tokenizer.fit_on_texts(pd.concat((x['question1'], x['question2'])).unique())

            q1_sequences = keras_tokenizer.texts_to_sequences(x['question1'])
            q2_sequences = keras_tokenizer.texts_to_sequences(x['question2'])
            q1_sequences = pad_sequences(q1_sequences, maxlen=max_features)
            q2_sequences = pad_sequences(q2_sequences, maxlen=max_features)
            labels = np.asarray(y)

            if mode == 'train':
                handle_pickle(text_to_seq_file, "write", (q1_sequences, q2_sequences, labels, keras_tokenizer))

        return q1_sequences, q2_sequences, labels, keras_tokenizer

    except Exception as e:
        raise Exception("Unable to generate text to sequences due to exception {}".format(e))


def get_custom_model(max_features, vocab_len, embedding_weights=None):
    """
    Method builds keras model to process quora duplicate question pair detection task.
    :param max_features: Input feature length
    :param vocab_len: Vocabulary length (used for embedding layer)
    :param embedding_weights: If not None, embedding layer is load with these weights.
                              Weighted are frozen before training. If None, embedding layer is trained from scratch.
                              Shape of embedding layer is (batch_size,sequence_length,300)
    :return: keras model
    """
    inp_q1 = Input(shape=(max_features,), dtype=np.int32)
    inp_q2 = Input(shape=(max_features,), dtype=np.int32)

    if embedding_weights is None:
        embed_layer = Embedding(input_dim=vocab_len, output_dim=300)
    else:
        embed_layer = Embedding(input_dim=vocab_len, output_dim=300, weights=[embedding_weights], trainable=False)

    q1 = embed_layer(inp_q1)
    q1 = LSTM(60)(q1)
    q1 = Dense(128, activation="relu")(q1)
    q1 = Dropout(0.2)(q1)
    q1 = BatchNormalization()(q1)
    q1 = Dense(64, activation="relu")(q1)
    q1 = Dropout(0.2)(q1)
    q1 = BatchNormalization()(q1)

    q2 = embed_layer(inp_q2)
    q2 = LSTM(60)(q2)
    q2 = Dense(128, activation="relu")(q2)
    q2 = Dropout(0.2)(q2)
    q2 = BatchNormalization()(q2)
    q2 = Dense(64, activation="relu")(q2)
    q2 = Dropout(0.2)(q2)
    q2 = BatchNormalization()(q2)

    output = concatenate([q1, q2])
    output = Dropout(0.3)(output)
    output = BatchNormalization()(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.4)(output)
    output = BatchNormalization()(output)
    output = Dense(1, activation='sigmoid')(output)

    model = keras.Model(inputs=[inp_q1, inp_q2], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def get_tf_hub_models(model_url):
    """
    Method builds keras model to process quora duplicate question pair detection task.
    The model will be built on top of tensorflow hub model loaded from model_url. The tensorflow hub model layers will
    be frozen before training.
    :param model_url: Tensorflow hub url
    :return: keras model
    """
    embed_layer = hub.KerasLayer(model_url, trainable=False)

    inp_q1 = Input(shape=[], dtype=tf.string)
    q1 = embed_layer(inp_q1)
    q1 = Dense(128, activation="relu")(q1)
    q1 = Dropout(0.2)(q1)
    q1 = BatchNormalization()(q1)
    q1 = Dense(64, activation="relu")(q1)
    q1 = Dropout(0.2)(q1)
    q1 = BatchNormalization()(q1)

    inp_q2 = Input(shape=[], dtype=tf.string)
    q2 = embed_layer(inp_q2)
    q2 = Dense(128, activation="relu")(q2)
    q2 = Dropout(0.2)(q2)
    q2 = BatchNormalization()(q2)
    q2 = Dense(64, activation="relu")(q2)
    q2 = Dropout(0.2)(q2)
    q2 = BatchNormalization()(q2)

    output = concatenate([q1, q2])
    output = Dropout(0.3)(output)
    output = BatchNormalization()(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.4)(output)
    output = BatchNormalization()(output)
    output = Dense(1, activation='sigmoid')(output)

    model = keras.Model(inputs=[inp_q1, inp_q2], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def get_glove_embeddings(keras_tokenizer):
    """
    Method generate embedding matrix weights from glove embeddings
    :param keras_tokenizer: keras_tokenizer to retrieve sequence vocabulary
    :return: embedding weights matrix
    """
    embeddings_index = {}
    f = open(glove_embedding_text)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(keras_tokenizer.word_index) + 1, 300))
    for word, i in keras_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def plot_accuracy(model_name, history):
    """
    Method to plot model training history
    :param model_name: model name
    :param history: history.history object containing accuracy, loss, val_loss and val_accuracy
                    returned by keras model fit method
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(model_name + ' - Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def print_model_summary(model_name, model):
    """
    Method to print keras model summary
    :param model_name: model name
    :param model: keras model
    """
    print("\n", model_name, "\n")
    print(model.summary())
    print("\n")


def train_model(model_name, x_train, y_train, x_val, y_val, epochs, batch_size, retrain=False, max_features=None,
                vocab_size=None, embedding_weights=None):
    """
    Method to train deep learning model. Model will be trained and saved under models_folder + model_name location.
    Also history.history object from model.fit method will be saved under
    models_folder + model_name + '_history' location
    :param model_name: Model to train. Should be in ['model_a','model_b','model_c','model_d']
    :param x_train: training features. Should be a list of features [x_q1,x_q2] for question 1 and question 2
    :param y_train: training labels
    :param x_val: validation features. Should be a list of features [x_q1,x_q2] for question 1 and question 2
    :param y_val: validation labels
    :param epochs: total epochs to train the model
    :param batch_size: batch size for training
    :param retrain: if True, model will be loaded and retrained on top of the existing model
    :param max_features: Input feature length
    :param vocab_size: Vocabulary length (used for embedding layer)
    :param embedding_weights: glove embedding weights matrix generated from get_glove_embeddings method
    """
    model = None
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    if retrain:
        model = keras.models.load_model(models_folder + model_name)
    else:
        if model_name == 'model_a':
            model = get_custom_model(max_features, vocab_size)

        elif model_name == 'model_b':
            model = get_custom_model(max_features, vocab_size, embedding_weights)

        elif model_name == 'model_c':
            model = get_tf_hub_models(c.d_universal_sentence_encoder)

        elif model_name == 'model_d':
            model = get_tf_hub_models(c.d_wiki_skip_gram_model)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        callbacks=[callback], shuffle=True)
    model.save(models_folder + model_name)
    with open(models_folder + model_name + '_history', "wb") as f:
        pickle.dump(history.history, f)


def load_model(model_name):
    """
    Method to load the trained model from models_folder + model_name location
    :param model_name: model name
    :return: loaded model
    """
    try:
        model = keras.models.load_model(models_folder + model_name)
        print_model_summary(model_name, model)
        with open(models_folder + model_name + '_history', "rb") as f:
            history = pickle.load(f)
            plot_accuracy(model_name, history)
        print(model_name, " model loaded")
        return model
    except Exception as e:
        raise Exception(model_name,
                        " model loading failed!! Please retrain by calling this method with overwrite = True")


def train_nn(epochs=20, batch_size=32, retrain=False, overwrite_data=False):
    """
    Method to train models for quora duplicate question pairs detection task. Four models will be trained on this task
    4 deep learning models - ['model_a','model_b','model_c','model_d'] (using embedding layers and tensorflow hub)
    The models will be stored under models_folder + model_name location
    :param epochs: total epochs to train the model
    :param batch_size: batch size for training
    :param retrain: if True, model will be loaded and retrained on top of the existing model
    :param overwrite_data: If set to True, preprocess files and feature files will be re processed.
                           Else data will be retrieved from existing files if exist. Need to set it to True when running
                           for first time else will throw file not found exception.
    """
    print("Deep learning models training started..")

    data = pd.read_csv(quora_duplicate_questions, sep='\t').dropna()
    rus = RandomUnderSampler(random_state=42, return_indices=False)
    data_x, data_y = rus.fit_sample(data[['question1', 'question2']], data[['is_duplicate']])
    data = pd.concat((pd.DataFrame(data=data_x, columns=['question1', 'question2']),
                      pd.DataFrame(data=data_y, columns=['is_duplicate'])), axis=1)
    del data_x, data_y
    print("data size :", data.shape)
    print("Labels category counts :")
    print(data['is_duplicate'].value_counts().to_string())
    preprocess_data = load_preprocessed_data('train', data, quora_pre_processed_file, overwrite_data)

    q1_sequences, q2_sequences, labels, keras_tokenizer = load_text_to_sequences('train', preprocess_data,
                                                                                 quora_text_to_seq_file, max_features,
                                                                                 overwrite_data)
    q1_train, _, q2_train, _, y_train_w, _ = train_test_split(q1_sequences, q2_sequences, labels.ravel(), test_size=0.2,
                                                              random_state=42)
    q1_train, q1_val, q2_train, q2_val, y_train_w, y_val_w = train_test_split(q1_train, q2_train, y_train_w,
                                                                              test_size=0.1, random_state=42)
    vocab_size = len(keras_tokenizer.word_index) + 1
    glove_embeddings = None

    for model_name in ['model_a', 'model_b']:
        if model_name == 'model_b':
            glove_embeddings = get_glove_embeddings(keras_tokenizer)
        train_model(model_name, [q1_train, q2_train], y_train_w, [q1_val, q2_val], y_val_w, epochs, batch_size, retrain,
                    max_features, vocab_size, glove_embeddings)

    del q1_sequences, q2_sequences, labels, keras_tokenizer, q1_train, q2_train, y_train_w, q1_val, q2_val, y_val_w
    del glove_embeddings

    x_train, _, y_train, _ = train_test_split(preprocess_data[0], preprocess_data[1], test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    for model_name in ['model_c', 'model_d']:
        train_model(model_name, [x_train['question1'], x_train['question2']], y_train,
                    [x_val['question1'], x_val['question2']], y_val, epochs, batch_size, retrain)

    del x_train, y_train, x_val, y_val, preprocess_data

    print("Deep learning models training complete...")


def train_rf_model(estimators=80, overwrite_data=False):
    """
    Method to train models for quora duplicate question pairs detection task. Two models will be trained on this task
    2 random forest models - ['rf_bow','rf_tfidf'] (using bag of words and tfidf features and no. of estimators = 80)
    The models will be stored under models_folder + model_name location
    :param estimators: total number of decision tree estimators
    :param overwrite_data: If set to True, preprocess files and feature files will be re processed.
                           Else data will be retrieved from existing files if exist. Need to set it to True when running
                           for first time else will throw file not found exception.
    """
    print("Training Random forest started..")

    preprocess_data = load_preprocessed_data('train', None, quora_pre_processed_file, False)

    bow_q1, bow_q2, bow_labels, count_vectorizer = load_bag_of_words_features('train', preprocess_data,
                                                                              quora_bag_of_words_file, rf_max_features,
                                                                              overwrite_data)
    rf_bow_x = np.concatenate((bow_q1, bow_q2), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(rf_bow_x, bow_labels.ravel(), test_size=0.2, random_state=42)
    rf_bow_model = RandomForestClassifier(n_estimators=estimators)
    rf_bow_model.fit(x_train, y_train)
    pickle.dump(rf_bow_model, open(models_folder + 'rf_bow', 'wb'))

    del bow_q1, bow_q2, bow_labels, count_vectorizer, rf_bow_x, x_train, x_test, y_train, y_test, rf_bow_model

    tfidf_q1, tfidf_q2, tfidf_labels, tfidf_vectorizer = load_tfidf_features('train', preprocess_data,
                                                                             quora_tf_idf_file, rf_max_features,
                                                                             overwrite_data)
    rf_tfidf_x = np.concatenate((tfidf_q1, tfidf_q2), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(rf_tfidf_x, tfidf_labels.ravel(), test_size=0.2,
                                                        random_state=42)
    rf_tfidf_model = RandomForestClassifier(n_estimators=estimators)
    rf_tfidf_model.fit(x_train, y_train)
    pickle.dump(rf_tfidf_model, open(models_folder + 'rf_tfidf', 'wb'))

    print("Training Random forest completed..")


def generate_results():
    """
    Method evaluates all the models generated from train method. A test set will be fetched from data source and
    evaluation will be performed against generated predictions.
    :return:  accuracy report containing model_name, accuracy, precision, recall and fl-score for each model
    """
    accuracy_report = []
    preprocess_data = load_preprocessed_data('train', None, quora_pre_processed_file, False)
    for model_name in ['model_a', 'model_b', 'model_c', 'model_d', 'rf_bow', 'rf_tfidf']:
        if model_name in ['rf_bow', 'rf_tfidf']:
            model = pickle.load(open(models_folder + model_name, 'rb'))
        else:
            model = load_model(model_name)
        y_test = None
        y_pred = None
        if model_name in ['model_a', 'model_b']:
            q1_sequences, q2_sequences, labels, _ = load_text_to_sequences('train', None, quora_text_to_seq_file,
                                                                           max_features, False)
            _, q1_test, _, q2_test, _, y_test = train_test_split(q1_sequences, q2_sequences, labels.ravel(),
                                                                 test_size=0.2, random_state=42)
            y_prob = model.predict([q1_test, q2_test])
            y_pred = tf.greater(y_prob, .5)
            del q1_sequences, q2_sequences, labels, q1_test, q2_test, y_prob, model
        elif model_name in ['model_c', 'model_d']:
            _, x_test, _, y_test = train_test_split(preprocess_data[0], preprocess_data[1], test_size=0.2,
                                                    random_state=42)
            y_prob = model.predict([x_test['question1'], x_test['question2']])
            y_pred = tf.greater(y_prob, .5)
            del x_test, y_prob, model
        elif model_name == 'rf_bow':
            bow_q1, bow_q2, bow_labels, _ = load_bag_of_words_features('train', preprocess_data,
                                                                       quora_bag_of_words_file, max_features, False)
            rf_bow_x = np.concatenate((bow_q1, bow_q2), axis=1)
            _, x_test, _, y_test = train_test_split(rf_bow_x, bow_labels.ravel(), test_size=0.2, random_state=42)
            y_pred = model.predict(x_test)
            del bow_q1, bow_q2, bow_labels, rf_bow_x, x_test, model
        elif model_name == 'rf_tfidf':
            tfidf_q1, tfidf_q2, tfidf_labels, _ = load_tfidf_features('train', preprocess_data, quora_tf_idf_file,
                                                                      max_features, False)
            rf_tfidf_x = np.concatenate((tfidf_q1, tfidf_q2), axis=1)
            _, x_test, _, y_test = train_test_split(rf_tfidf_x, tfidf_labels.ravel(), test_size=0.2, random_state=42)
            y_pred = model.predict(x_test)
            del tfidf_q1, tfidf_q2, tfidf_labels, rf_tfidf_x, x_test, model

        accuracy_report.append([model_name,
                                format(accuracy_score(y_test, y_pred) * 100, '.2f'),
                                format(precision_score(y_test, y_pred) * 100, '.2f'),
                                format(recall_score(y_test, y_pred) * 100, '.2f'),
                                format(f1_score(y_test, y_pred) * 100, '.2f')])
    return accuracy_report


def print_results(accuracy_report):
    """
    Method to print the accuracy report using pretty table module
    :param accuracy_report: accuracy report from generate_results method
    """
    print("Testing results:")
    t = PrettyTable(['Model', 'Acuracy', 'Precision', 'Recall', 'F1-Score'])
    for obj in accuracy_report:
        t.add_row([obj[0], obj[1], obj[2], obj[3], obj[4]])
    print(t)


if __name__ == '__main__':
    # Train deep learning models
    train_nn(epochs=10, batch_size=128, retrain=False, overwrite_data=True)

    # Train Random Forest models
    train_rf_model(estimators=80, overwrite_data=True)

    # Generate predictions and evaluate
    d_accuracy_report = generate_results()

    # Print the results
    print_results(d_accuracy_report)
