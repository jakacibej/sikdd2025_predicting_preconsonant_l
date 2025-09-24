from collections import defaultdict as dd
import os
from scipy.stats import chi2_contingency
from scipy.stats.contingency import margins
import numpy as np

sep = " ~~~ "

# https://www.geeksforgeeks.org/python/python-pearsons-chi-square-test/
# https://www.geeksforgeeks.org/python/how-to-calculate-cramers-v-in-python/
# https://stackoverflow.com/questions/56610686/calculate-pearsons-standardized-residuals-in-python

def residuals(observed, expected):
    return (observed - expected) / np.sqrt(expected)

def stdres(observed, expected):
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    return (observed - expected) / np.sqrt(v)

class PreconsonantLTool:


    def __init__(self):
        self.dict_to_convert_to_CVC_robust = dd()
        self.dict_to_convert_to_CVC_finegrained = dd()
        self.list_of_consonants = []
        self.list_of_vowels = []


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


    def extract_ngram_surroundings_of_preconsonant_l(self, string, index_of_preconsonant_l_occurrence, max_ngram_length):

        characters_in_string = list(string)

        characters_before_preconsonant_l = characters_in_string[:index_of_preconsonant_l_occurrence]
        characters_after_preconsonant_l = characters_in_string[index_of_preconsonant_l_occurrence+1:]

        all_subsequent_ngrams = []
        for i in range(0, max_ngram_length):
            if i+1 > len(characters_after_preconsonant_l):
                break
            try:
                subsequent_ngram = "".join(characters_after_preconsonant_l[0:i+1])
                if not subsequent_ngram in [""]:
                    all_subsequent_ngrams.append(subsequent_ngram)
            except:
                continue

        all_preceding_ngrams = []
        for i in range(0, max_ngram_length):
            try:
                preceding_ngram = "".join(characters_before_preconsonant_l[(-i-1):])
                if i >= len(characters_before_preconsonant_l):
                    break
                if not preceding_ngram in [""]:
                    all_preceding_ngrams.append(preceding_ngram)
            except:
                continue

        return [x for x in reversed(all_preceding_ngrams)], all_subsequent_ngrams


# INSTANTIATE PRECONSONANT L EXTRACTION TOOL
preconsonant_l_tool_instance = PreconsonantLTool()

# DICTIONARIES TO COLLECT n-GRAMS
dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation = dd(lambda: dd(int))
dict_GENERAL_SUBSEQUENT_NGRAMS_and_pronunciation = dd(lambda: dd(int))
dict_CVC_ROBUST_PRECEDING_NGRAMS_and_pronunciation = dd(lambda: dd(int))
dict_CVC_ROBUST_SUBSEQUENT_NGRAMS_and_pronunciation = dd(lambda: dd(int))
dict_CVC_FINEGRAINED_PRECEDING_NGRAMS_and_pronunciation = dd(lambda: dd(int))
dict_CVC_FINEGRAINED_SUBSEQUENT_NGRAMS_and_pronunciation = dd(lambda: dd(int))

#total_number_of_forms_with_complete_agreement_in_ils_1_0 = 0  # TO COUNT FORMS THAT WE TAKE INTO ACCOUNT FOR MACHINE LEARNING
total_number_of_preconsonant_l_with_complete_agreement_in_ils_1_0 = 0

#dict_COUNT_ALL_FORMS_by_PRONUNCIATION = dd(int)
dict_COUNT_ALL_PRECONSONANT_L_OCCURENCES_by_PRONUNCIATION = dd(int)

# TODO - SETTINGS
maximum_ngram_length = 5  # SET THE MAXIMUM LENGTH OF n-GRAMS TO EXTRACT FROM THE SURROUNDINGS OF A PRE-CONSONANT L




# ILS 1.0 DATASET
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

        #total_number_of_forms_with_complete_agreement_in_ils_1_0 += 1
        #dict_COUNT_ALL_FORMS_by_PRONUNCIATION[collective_response] += 1

        # GET A WORD
        string_ORIGINAL = form
        #string_ORIGINAL = "Gledalka"
        #string_ORIGINAL = "PREBivaLstven"
        #string_ORIGINAL = "AlKin"

        # CONVERT THE WORD TO LOWERCASE
        string_LOWERCASE = string_ORIGINAL.lower()

        # GET CVC FORMS OF THE SELECTED WORD
        string_CVC_ROBUST = preconsonant_l_tool_instance.convert_to_CVC_robust(string=string_LOWERCASE)
        string_CVC_FINEGRAINED = preconsonant_l_tool_instance.convert_to_CVC_finegrained(string=string_LOWERCASE)

        # GET FORMS OF THE SELECTED WORD WITH WORD BOUNDARY MARKERS (these are used in n-gram extraction)
        string_LOWERCASE_with_word_boundary_markers = f"#{string_LOWERCASE}#"
        string_CVC_ROBUST_with_word_boundary_markers = f"#{string_CVC_ROBUST}#"
        string_CVC_FINEGRAINED_with_word_boundary_markers = f"#{string_CVC_FINEGRAINED}#"

        #print(string_ORIGINAL, string_LOWERCASE, string_LOWERCASE_with_word_boundary_markers)

        # COLLECT A LIST OF INDICES WHERE THE PRECONSONANT L OCCURS IN THE WORD
        list_of_indices_with_preconsonant_l = preconsonant_l_tool_instance.return_list_of_indices_with_preconsonant_l_occurrences_in_string(string=string_LOWERCASE_with_word_boundary_markers)
        # FOR EACH INDEX, EXTRACT ITS SURROUNDING n-GRAMS
        for index in list_of_indices_with_preconsonant_l:
            #print(preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_LOWERCASE_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length))
            #print(preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_ROBUST_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length))
            #print(preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_FINEGRAINED_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length))

            total_number_of_preconsonant_l_with_complete_agreement_in_ils_1_0 += 1
            dict_COUNT_ALL_PRECONSONANT_L_OCCURENCES_by_PRONUNCIATION[collective_response] += 1

            GENERAL_preceding_ngrams, GENERAL_subsequent_ngrams = preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_LOWERCASE_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length)
            CVC_ROBUST_preceding_ngrams, CVC_ROBUST_subsequent_ngrams = preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_ROBUST_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length)
            CVC_FINEGRAINED_preceding_ngrams, CVC_FINEGRAINED_subsequent_ngrams = preconsonant_l_tool_instance.extract_ngram_surroundings_of_preconsonant_l(string=string_CVC_FINEGRAINED_with_word_boundary_markers, index_of_preconsonant_l_occurrence=index, max_ngram_length=maximum_ngram_length)

            # POPULATE DICT WITH GENERAL n-GRAMS
            for ngram in GENERAL_preceding_ngrams:
                dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation[ngram][collective_response] += 1
            for ngram in GENERAL_subsequent_ngrams:
                dict_GENERAL_SUBSEQUENT_NGRAMS_and_pronunciation[ngram][collective_response] += 1

            # POPULATE DICT WITH CVC ROBUST n-GRAMS
            for ngram in CVC_ROBUST_preceding_ngrams:
                dict_CVC_ROBUST_PRECEDING_NGRAMS_and_pronunciation[ngram][collective_response] += 1
            for ngram in CVC_ROBUST_subsequent_ngrams:
                dict_CVC_ROBUST_SUBSEQUENT_NGRAMS_and_pronunciation[ngram][collective_response] += 1

            # POPULATE DICT WITH CVC FINEGRAINED n-GRAMS
            for ngram in CVC_FINEGRAINED_preceding_ngrams:
                dict_CVC_FINEGRAINED_PRECEDING_NGRAMS_and_pronunciation[ngram][collective_response] += 1
            for ngram in CVC_FINEGRAINED_subsequent_ngrams:
                dict_CVC_FINEGRAINED_SUBSEQUENT_NGRAMS_and_pronunciation[ngram][collective_response] += 1


# GENERATE CONTINGENCY TABLES FOR EXTRACTED n-GRAMS
#relevant_dictionary = dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation
#relevant_dictionary = dict_GENERAL_SUBSEQUENT_NGRAMS_and_pronunciation
#relevant_dictionary = dict_CVC_ROBUST_PRECEDING_NGRAMS_and_pronunciation
#relevant_dictionary = dict_CVC_ROBUST_SUBSEQUENT_NGRAMS_and_pronunciation
#relevant_dictionary = dict_CVC_FINEGRAINED_PRECEDING_NGRAMS_and_pronunciation
relevant_dictionary = dict_CVC_FINEGRAINED_SUBSEQUENT_NGRAMS_and_pronunciation

for file, relevant_dictionary in [("chi_square_GENERAL_PRECEDING_ngrams.tsv", dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation),
                                  ("chi_square_GENERAL_SUBSEQUENT_ngrams.tsv", dict_GENERAL_SUBSEQUENT_NGRAMS_and_pronunciation),
                                  ("chi_square_CVC-ROBUST_PRECEDING_ngrams.tsv", dict_CVC_ROBUST_PRECEDING_NGRAMS_and_pronunciation),
                                  ("chi_square_CVC-ROBUST_SUBSEQUENT_ngrams.tsv", dict_CVC_ROBUST_SUBSEQUENT_NGRAMS_and_pronunciation),
                                  ("chi_square_CVC-FINEGRAINED_PRECEDING_ngrams.tsv", dict_CVC_FINEGRAINED_PRECEDING_NGRAMS_and_pronunciation),
                                  ("chi_square_CVC-FINEGRAINED_SUBSEQUENT_ngrams.tsv", dict_CVC_FINEGRAINED_SUBSEQUENT_NGRAMS_and_pronunciation)
                                  ]:

    output_file = open(file, "w", encoding="UTF-8")
    output_file.write("{}\n".format("\t".join([str(x) for x in ["ngram", "chi_square_statistic", "p_value", "robust_p_value", "df", "N", "minimum_dimension", "Cramers_V", "max_residual", "residual_interpretation"]])))


    counter = 0

    for ngram in relevant_dictionary:
        data_for_contingency_table = []

        for response in [f"L{sep}L", f"U{sep}U", f"B{sep}B"]:
            data_for_individual_response = []
            data_for_individual_response.append(relevant_dictionary[ngram][response])
            #data_for_individual_response.append(dict_COUNT_ALL_FORMS_by_PRONUNCIATION[response]-relevant_dictionary[ngram][response])
            data_for_individual_response.append(dict_COUNT_ALL_PRECONSONANT_L_OCCURENCES_by_PRONUNCIATION[response] - relevant_dictionary[ngram][response])

            data_for_contingency_table.append(data_for_individual_response)

            #print(ngram, dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation[ngram][response], dict_COUNT_ALL_FORMS_by_PRONUNCIATION[response]-dict_GENERAL_PRECEDING_NGRAMS_and_pronunciation[ngram][response])

        #print(ngram)
        #print(data_for_contingency_table)
        #print(data_for_contingency_table)
        data_for_contingency_table = np.array(data_for_contingency_table)
        #print(ngram)
        #print(data_for_contingency_table)
        try:
            chi_square_statistic, p_value, dof, expected = chi2_contingency(observed=data_for_contingency_table)
        except:
            print("UNABLE TO CALCULATE CHI-SQUARE TEST for n-GRAM", ngram)
            continue

        N = np.sum(data_for_contingency_table)
        minimum_dimension = min(data_for_contingency_table.shape)-1

        # https://www.geeksforgeeks.org/python/how-to-calculate-cramers-v-in-python/
        # √(X2/N) / min(C-1, R-1)
        #
        #
        # Here,
        #
        #
        # X2: It is the Chi-square statistic
        # N: It represents the total sample size
        # R: It is equal to the number of rows
        # C: It is equal to the number of columns
        Cramers_V = np.sqrt( (chi_square_statistic/N) / minimum_dimension)

        #The p-values should be interpreted in the following manner: **** → p ≤ 0.0001; *** → p ≤ 0.001; ** → p ≤ 0.01; * → p ≤ 0.05

        if p_value <= 0.0001:
            robust_p_value = "****"
        elif p_value <= 0.001:
            robust_p_value = "***"
        elif p_value <= 0.01:
            robust_p_value = "**"
        elif p_value <= 0.05:
            robust_p_value = "*"
        elif p_value > 0.05:
            robust_p_value = "NS"
        else:
            robust_p_value = "_"

        # Pearson's residuals
        #Pearson_residuals = stdres(observed=data_for_contingency_table, expected=expected)
        Pearson_residuals = residuals(observed=data_for_contingency_table, expected=expected)
        #print(Pearson_residuals)
        list_of_Pearson_residuals_and_their_positions = []
        count_row = 0
        for presidual in Pearson_residuals:
            count_row += 1
            count_column = 0
            for subelement in presidual:
                count_column += 1
                list_of_Pearson_residuals_and_their_positions.append((subelement, count_row, count_column))

        maximum_absolute_value_of_Pearson_residuals = max([abs(x[0]) for x in list_of_Pearson_residuals_and_their_positions])
        #print(ngram, "MAX ABSOLUTE RESIDUAL:", maximum_absolute_value_of_Pearson_residuals)
        for residual_tuple in list_of_Pearson_residuals_and_their_positions:
            if abs(residual_tuple[0]) == maximum_absolute_value_of_Pearson_residuals:
                row = residual_tuple[1]
                column = residual_tuple[2]

                true_value_of_maximum_residual = residual_tuple[0]

                # ROW 1 = L
                # ROW 2 = U
                # ROW 3 = B
                # COLUMN 1 = present in surrounding of pre-consonant l
                # COLUMN 2 = absent from surrounding of pre-consonant l
                # INTERPRET THE RESIDUAL
                if (column, row) == (1, 1):
                    residual_interpretation = "in surroundings for L pronunciation"
                elif (column, row) == (1, 2):
                    residual_interpretation = "not in surroundings for L pronunciation"
                elif (column, row) == (2, 1):
                    residual_interpretation = "in surroundings for U pronunciation"
                elif (column, row) == (2, 2):
                    residual_interpretation = "not in surroundings for U pronunciation"
                elif (column, row) == (3, 1):
                    residual_interpretation = "in surroundings for B pronunciation"
                elif (column, row) == (3, 2):
                    residual_interpretation = "not in surroundings for B pronunciation"


        if ngram == "c":
            print(data_for_contingency_table)
            print(Pearson_residuals)


        output_file.write("{}\n".format("\t".join([str(x) for x in [ngram, chi_square_statistic, p_value, robust_p_value, dof, N, minimum_dimension, Cramers_V, true_value_of_maximum_residual, residual_interpretation]])))

        #if p_value <= 0.05:
        #if p_value > 0.05:
        #    print(ngram, chi_square_statistic, p_value, dof, Cramers_V)
        #    counter += 1

    print(counter)




"""
žvižgalka

#žvižgalka#

#žvižga l ka#

Features:

Preceding 1-grams: a
Preceding 2-grams: ga
Preceding 3-grams: žga
Preceding 4-grams: ižga

Preceding robust CV 1-grams: V
Preceding robust CV 2-grams: CV
Preceding robust CV 3-grams: CCV
Preceding robust CV 4-grams: VCCV

Preceding finegrained CV 1-grams: V
Preceding finegrained CV 2-grams: GV
Preceding finegrained CV 3-grams: GGV
Preceding finegrained CV 4-grams: VGGV

Subsequent 1-grams: k
Subsequent 2-grams: ka
Subsequent 3-grams: ka#
Subsequent 4-grams: -

Subsequent robust CV 1-grams: C
Subsequent robust CV 2-grams: CV
Subsequent robust CV 3-grams: CV#
Subsequent robust CV 4-grams: -

Subsequent finegrained CV 1-grams: K
Subsequent finegrained CV 2-grams: KV
Subsequent finegrained CV 3-grams: KV#
Subsequent finegrained CV 4-grams: -
"""