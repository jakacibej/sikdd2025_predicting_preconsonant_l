import os
from collections import defaultdict as dd

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import dummy
import numpy as np
from sklearn.model_selection import cross_val_score
import random
from sklearn import preprocessing
import pickle



sep = " ~~~ "


class PreconsonantLVectorizer:

    def __init__(self):
        self.dict_to_convert_to_CVC_robust = dd()
        self.dict_to_convert_to_CVC_finegrained = dd()
        self.list_of_consonants = []
        self.list_of_vowels = []

        self.list_of_ngrams_general_preceding = [x.strip("\n") for x in open(f"./resources/02_ngrams_general_preceding.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_ngrams_general_subsequent = [x.strip("\n") for x in open(f"./resources/01_ngrams_general_subsequent.tsv", "r", encoding="UTF-8").readlines()]

        self.list_of_ngrams_cvc_robust_preceding = [x.strip("\n") for x in open(f"./resources/04_ngrams_cvc-robust_preceding.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_ngrams_cvc_robust_subsequent = [x.strip("\n") for x in open(f"./resources/03_ngrams_cvc-robust_subsequent.tsv", "r", encoding="UTF-8").readlines()]

        self.list_of_ngrams_cvc_finegrained_preceding = [x.strip("\n") for x in open(f"./resources/06_ngrams_cvc-finegrained_preceding.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_ngrams_cvc_finegrained_subsequent = [x.strip("\n") for x in open(f"./resources/05_ngrams_cvc-finegrained_subsequent.tsv", "r", encoding="UTF-8").readlines()]

        self.max_ngram_length = 5  # The statistical analysis of n-grams took into account character-level n-grams (1 <= n <= 5)

        # POPULATE LIST OF MORPHOSYNTACTIC FEATURES
        self.list_of_mte6_morphosyntactic_feature_tuples = []
        for line in open(f"./resources/mte6_feature_table.tsv", "r", encoding="UTF-8").readlines()[1:]:  # SKIP HEADERS
            POS_sl, POS_en, Feature_Name, Position_in_MSD, Value_sl, Value_en = line.strip("\n").split("\t")
            self.list_of_mte6_morphosyntactic_feature_tuples.append((POS_sl, Position_in_MSD, Value_sl))

        # COMPILE DICTIONARIES FOR CVC-CONVERSION
        for file in os.listdir('./resources'):
            if file in ["GF2.0_character_categorization.txt"]:
                for line in open(f"./resources/{file}", "r", encoding="UTF-8").readlines()[1:]:  # SKIP HEADERS
                    character_string,\
                    CV_category,\
                    CV_category_finegrained = line.strip("\n").split("\t")

                    self.dict_to_convert_to_CVC_robust[character_string] = CV_category
                    self.dict_to_convert_to_CVC_finegrained[character_string] = CV_category_finegrained

                    if CV_category in ["C"]:
                        self.list_of_consonants.append(character_string)
                    elif CV_category in ["V"]:
                        self.list_of_vowels.append(character_string)

    def convert_to_CVC_robust(self, string):
        converted_characters = []
        for character in list(string):
            if character in self.dict_to_convert_to_CVC_robust:
                converted_characters.append(self.dict_to_convert_to_CVC_robust[character])
            else:
                converted_characters.append("-")

        return "".join(converted_characters)


    def convert_to_CVC_finegrained(self, string):
        converted_characters = []
        for character in list(string):
            if character in self.dict_to_convert_to_CVC_finegrained:
                converted_characters.append(self.dict_to_convert_to_CVC_finegrained[character])
            else:
                converted_characters.append("-")

        return "".join(converted_characters)


    def return_list_of_indices_with_preconsonant_l_occurrences_in_string(self, string):
        indices_for_preconsonant_l_occurrences = []

        characters_in_lowercase_string = list(string.lower())
        for index, character in enumerate(characters_in_lowercase_string):
            try:
                if character in ["l"] and characters_in_lowercase_string[index+1] in self.list_of_consonants:
                    indices_for_preconsonant_l_occurrences.append(index)
            except:
                continue

        return indices_for_preconsonant_l_occurrences


    def extract_ngram_surroundings_of_preconsonant_l(self, string, index_of_preconsonant_l_occurrence):

        characters_in_string = list(string)

        characters_before_preconsonant_l = characters_in_string[:index_of_preconsonant_l_occurrence]
        characters_after_preconsonant_l = characters_in_string[index_of_preconsonant_l_occurrence+1:]

        all_subsequent_ngrams = []
        for i in range(0, self.max_ngram_length):
            if i+1 > len(characters_after_preconsonant_l):
                break
            try:
                subsequent_ngram = "".join(characters_after_preconsonant_l[0:i+1])
                if not subsequent_ngram in [""]:
                    all_subsequent_ngrams.append(subsequent_ngram)
            except:
                continue

        all_preceding_ngrams = []
        for i in range(0, self.max_ngram_length):
            try:
                preceding_ngram = "".join(characters_before_preconsonant_l[(-i-1):])
                if i >= len(characters_before_preconsonant_l):
                    break
                if not preceding_ngram in [""]:
                    all_preceding_ngrams.append(preceding_ngram)
            except:
                continue

        return [x for x in reversed(all_preceding_ngrams)], all_subsequent_ngrams

    def vectorize_preconsonant_l(self, string, mte6_sl, index_of_preconsonant_l):

        # CONVERT THE WORD TO LOWERCASE
        string_LOWERCASE = string.lower()

        # GET CVC FORMS OF THE SELECTED WORD
        string_CVC_ROBUST = self.convert_to_CVC_robust(string=string_LOWERCASE)
        string_CVC_FINEGRAINED = self.convert_to_CVC_finegrained(string=string_LOWERCASE)

        # GET FORMS OF THE SELECTED WORD WITH WORD BOUNDARY MARKERS (these are used in n-gram extraction)
        string_LOWERCASE_with_word_boundary_markers = f"#{string_LOWERCASE}#"
        string_CVC_ROBUST_with_word_boundary_markers = f"#{string_CVC_ROBUST}#"
        string_CVC_FINEGRAINED_with_word_boundary_markers = f"#{string_CVC_FINEGRAINED}#"

        GENERAL_preceding_ngrams, GENERAL_subsequent_ngrams = self.extract_ngram_surroundings_of_preconsonant_l(string=string_LOWERCASE_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index_of_preconsonant_l)
        CVC_ROBUST_preceding_ngrams, CVC_ROBUST_subsequent_ngrams = self.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_ROBUST_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index_of_preconsonant_l)
        CVC_FINEGRAINED_preceding_ngrams, CVC_FINEGRAINED_subsequent_ngrams = self.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_FINEGRAINED_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index_of_preconsonant_l)


        numeric_vector = []

        # FOR EACH LIST, APPEND VALUES TO NUMERIC VECTOR
        for list_of_ngrams, extracted_ngrams in [(self.list_of_ngrams_general_preceding, GENERAL_preceding_ngrams),
                                                 (self.list_of_ngrams_general_subsequent, GENERAL_subsequent_ngrams),
                                                 (self.list_of_ngrams_cvc_robust_preceding, CVC_ROBUST_preceding_ngrams),
                                                 (self.list_of_ngrams_cvc_robust_subsequent, CVC_ROBUST_subsequent_ngrams),
                                                 (self.list_of_ngrams_cvc_finegrained_preceding, CVC_FINEGRAINED_preceding_ngrams),
                                                 (self.list_of_ngrams_cvc_finegrained_subsequent, CVC_FINEGRAINED_subsequent_ngrams)]:
            for ngram in list_of_ngrams:
                if ngram in extracted_ngrams:
                    numeric_vector.append(1)
                else:
                    numeric_vector.append(0)

        # TODO - FOR EACH MULTEXT-EAST v6 MORPHOSYNTACTIC FEATURE, APPEND VALUES TO NUMERIC VECTOR
        for morphosyntactic_feature_tuple in self.list_of_mte6_morphosyntactic_feature_tuples:
            POS_sl, position_of_morphosyntactic_feature, value_sl = morphosyntactic_feature_tuple

            list_of_elements_in_mte6_sl = list(mte6_sl)
            part_of_speech = list_of_elements_in_mte6_sl[0]

            try:
                # presence_of_morphosyntactic_feature
                if list_of_elements_in_mte6_sl[int(position_of_morphosyntactic_feature)] == value_sl and POS_sl == part_of_speech:
                    #print(value_sl, string)
                    numeric_vector.append(1)
                else:
                    numeric_vector.append(0)
            except:
                numeric_vector.append(0)

        return numeric_vector



# INSTANTIATE VECTORIZER
preconsonant_l_vectorizer = PreconsonantLVectorizer()

# VECTORS FOR TRAINING MODELS
list_of_pronunciation_categories = []
list_of_feature_vectors = []
list_of_forms_and_linguistic_data = []


# ILS 1.0 DATASET
counter_for_vectorization_process = 0
ils_1_0_tsv = open("d:\\PycharmProjects\\PyBossa_agreement\\00_Resources\\ILS_1.0\\ILS_1.0.tsv", "r", encoding="UTF-8").readlines()
for line in ils_1_0_tsv[1:]:  # SKIP HEADERS
    task_id,\
    sloleks_lexeme_id,\
    form,\
    lemma,\
    mte6_msd_sl,\
    lc_bigram,\
    word_ending,\
    response_1,\
    response_2,\
    username_1,\
    user_id_1,\
    comment_1,\
    username_2,\
    user_id_2,\
    comment_2,\
    task_duration_1,\
    task_duration_2 = line.strip("\n").split("\t")

    collective_response = f"{response_1}{sep}{response_2}"

    if collective_response in [f"L{sep}L", f"B{sep}B", f"U{sep}U"]:  # ONLY TAKE FORMS WITH COMPLETE AGREEMENT INTO ACCOUNT
        # COLLECT A LIST OF INDICES WHERE THE PRECONSONANT L OCCURS IN THE WORD
        list_of_indices_with_preconsonant_l = preconsonant_l_vectorizer.return_list_of_indices_with_preconsonant_l_occurrences_in_string(string=f"#{form.lower()}#")
        # FOR EACH INDEX, EXTRACT ITS SURROUNDING n-GRAMS AND VECTORIZE IT
        for index in list_of_indices_with_preconsonant_l:
            vector = preconsonant_l_vectorizer.vectorize_preconsonant_l(string=f"#{form.lower()}#", index_of_preconsonant_l=index, mte6_sl=mte6_msd_sl)

            category = collective_response.split(sep)[0]

            #print(category, len(vector))

            list_of_pronunciation_categories.append(category)
            list_of_feature_vectors.append(vector)
            list_of_forms_and_linguistic_data.append(f"{sloleks_lexeme_id}{sep}{form}{sep}{lemma}{sep}{mte6_msd_sl}")

            counter_for_vectorization_process += 1
            if counter_for_vectorization_process % 5000 == 0:
                print(f"Processed {counter_for_vectorization_process} vectorizations...")


# SPLIT TRAINING SET
# STRATIFY SAMPLE: https://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
print("Splitting training set...")
random.seed(1234)
data_train, data_test, y_train, y_true, lemfpos_train, lemfpos_test = train_test_split(list_of_feature_vectors, list_of_pronunciation_categories, list_of_forms_and_linguistic_data, test_size=0.2, stratify=list_of_pronunciation_categories)




# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# LINEAR SUPPORT VECTOR CLASSIFICATION
print("Training model...")
#classifier = LinearSVC()
#model = classifier.fit(data_train, y_train)
#name = "LinearSVC"

# kNN - KNeighborsClassifier
# n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None
#classifier = KNeighborsClassifier()
#model = classifier.fit(data_train, y_train)
#name = "kNN-5"

# kNN - KNeighborsClassifier
# n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None
classifier = KNeighborsClassifier(n_neighbors=10)
model = classifier.fit(data_train, y_train)
name = "kNN-10"

# MultinomialNB - Multinomial Naive Bayes
#classifier = MultinomialNB()
#model = classifier.fit(data_train, y_train)
#name = "Multinomial-NB"

# Random Forest Classifier
#classifier = RandomForestClassifier()
#model = classifier.fit(data_train, y_train)
#name = "RFC"

# PERFORM 10-FOLD CROSS-VALIDATION
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://stackoverflow.com/questions/27357121/scikit-calculate-precision-and-recall-using-cross-val-score-function
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
lb = preprocessing.LabelBinarizer()  # DATA NEEDS TO BE BINARIZED TO CALCULATE PRECISION ETC.

# TODO - SUPPRESS WARNINGS
import warnings
warnings.filterwarnings("ignore")


print("Performing crossvalidation...")
output_with_crossvalidation_scores = open(f"crossvalidation_scores_{name}.txt", "w", encoding="UTF-8")
#for score_type in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'precision_samples', 'recall_samples', 'f1_samples', 'roc_auc']:
for score_type in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:

    if not score_type in ['accuracy', 'balanced_accuracy']:  # FOR PRECISION, RECALL ETC., YOU NEED TO BINARIZE DATA
        #lb.fit(y_train)
        binarized_y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
        random.seed(1234)
        scores = cross_val_score(classifier, data_train, binarized_y_train, cv=10, scoring=score_type)
        print(score_type, scores, scores.mean(), scores.std())
        output_with_crossvalidation_scores.write("{}\n".format("\t".join([str(x) for x in [score_type, " | ".join([str(x) for x in scores]), scores.mean(), scores.std()]])))
    else:
        random.seed(1234)
        scores = cross_val_score(classifier, data_train, y_train, cv=10, scoring=score_type)
        print(score_type, scores, scores.mean(), scores.std())
        output_with_crossvalidation_scores.write("{}\n".format("\t".join([str(x) for x in [score_type, " | ".join([str(x) for x in scores]), scores.mean(), scores.std()]])))
        pass






y_test = model.predict(np.array(data_test))
output_with_predictions = open(f"predictions_{name}.txt", "w", encoding="UTF-8")
output_with_predictions.write("{}\n".format("\t".join(["true_label", "predicted_label", "lemfpos_info"])))
for i in zip(y_true, y_test, lemfpos_test):
    output_with_predictions.write("{}\n".format("\t".join([str(x) for x in i])))


baseline_classifier = dummy.DummyClassifier(strategy="most_frequent")
baseline_classifier.fit(data_train, y_train)
baseline_predictions = baseline_classifier.predict(data_train)
baseline_score = baseline_classifier.score(data_train, y_train)

print("|ACCURACY|{}|".format(metrics.accuracy_score(y_true, y_test)))
print("|PRECISION|{}|".format(metrics.precision_score(y_true, y_test, pos_label="U", average="macro")))
print("|RECALL|{}|".format(metrics.recall_score(y_true, y_test, pos_label="U", average="macro")))
print("|F1-SCORE|{}|".format(metrics.f1_score(y_true, y_test, pos_label="U", average="macro")))
print("|BASELINE - MOST FREQUENT|{}|".format(baseline_score))

output_with_crossvalidation_scores.write("### MAJORITY CLASSIFIER ###\n")
output_with_crossvalidation_scores.write("|ACCURACY: y_true vs. y_test|{}|\n".format(metrics.accuracy_score(y_true, y_test)))
output_with_crossvalidation_scores.write("|PRECISION: y_true vs. y_test; pos_label='U'|{}|\n".format(metrics.precision_score(y_true, y_test, pos_label="U", average='macro')))
output_with_crossvalidation_scores.write("|RECALL: y_true vs. y_test; pos_label='U'|{}|\n".format(metrics.recall_score(y_true, y_test, pos_label="U", average='macro')))
output_with_crossvalidation_scores.write("|F1-SCORE: y_true vs. y_test; pos_label='U'|{}|\n".format(metrics.f1_score(y_true, y_test, pos_label="U", average='macro')))
output_with_crossvalidation_scores.write("|BASELINE ACCURACY - MOST FREQUENT|{}|\n".format(baseline_score))

#X_train = ngram_counter.fit_transform(data_train)
#X_test  = ngram_counter.transform(data_test)


# SAVE MODELS
pickle.dump(model, open(f'{name}_model_export.sav', 'wb'))
