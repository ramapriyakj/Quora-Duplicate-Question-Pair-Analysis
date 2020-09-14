
# Project artifacts location
project_folder = 'D:/quora_questions_analysis/project_artifacts/'

# Source data location
quora_duplicate_questions = project_folder+'quora_duplicate_questions.tsv'


# Quora questions cluster analysis configuration
# ------------------------------------------------------------------------------
# Location to store intermediate project files
c_quora_pre_processed_file = project_folder+'quora_cluster_pre_processed'
c_quora_bag_of_words_file = project_folder+'quora_cluster_simple_bag_of_words'
c_quora_tf_idf_file = project_folder+'quora_cluster_tf_idf'
c_quora_lda_bow_file = project_folder+'quora_cluster_lda_bow'
c_quora_lda_tfidf_file = project_folder+'quora_cluster_lda_tfidf'

# Maximum features to generate for each question
c_max_features = 50
# ------------------------------------------------------------------------------


# Quora duplicate question pairs detection configuration
# ------------------------------------------------------------------------------
# Glove Embedding file location
# Download the glove embedding glove.6B.300d.txt from http://nlp.stanford.edu/data/glove.6B.zip and place it in the
# project folder and update the below config with the file location
d_glove_embedding_text = project_folder+'glove.6B.300d.txt'

# Location to store generated models
d_models_folder = project_folder+'models/'

# Location to store intermediate project files
d_quora_pre_processed_file = project_folder+'quora_pre_processed.pre'
d_quora_bag_of_words_file = project_folder+'quora_simple_bag_of_words.bow'
d_quora_tf_idf_file = project_folder+'quora_tf_idf.tfidf'
d_quora_text_to_seq_file = project_folder+'text_to_seq.seq'

# Maximum keras tokenizer features to generate for each question
d_max_features = 300

# Maximum bow and tfidf features to generate for each question
d_rf_max_features = 200

# Tensorflow hub model URLs
d_universal_sentence_encoder = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
d_wiki_skip_gram_model = 'https://tfhub.dev/google/Wiki-words-500-with-normalization/2'
# ------------------------------------------------------------------------------



