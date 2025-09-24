
for file in ["chi_square_CVC-FINEGRAINED_PRECEDING_ngrams_AFTER_HOLM-BONFERRONI.tsv",
             "chi_square_CVC-FINEGRAINED_SUBSEQUENT_ngrams_AFTER_HOLM-BONFERRONI.tsv",
             "chi_square_CVC-ROBUST_PRECEDING_ngrams_AFTER_HOLM-BONFERRONI.tsv",
             "chi_square_CVC-ROBUST_SUBSEQUENT_ngrams_AFTER_HOLM-BONFERRONI.tsv",
             "chi_square_GENERAL_PRECEDING_ngrams_AFTER_HOLM-BONFERRONI.tsv",
             "chi_square_GENERAL_SUBSEQUENT_ngrams_AFTER_HOLM-BONFERRONI.tsv"]:

    read_file = open(file, "r", encoding="UTF-8").readlines()

    output_file = open(file.replace('.tsv', '_feature_list.tsv'), "w", encoding="UTF-8")

    count_significant = 0

    for line in read_file[1:]:  # SKIP HEADERS
        ngram,\
        chi_square_statistic,\
        p_value,\
        robust_p_value,\
        df,\
        N,\
        minimum_dimension,\
        Cramers_V,\
        max_residual,\
        residual_interpretation,\
        alpha,\
        k,\
        m,\
        passes_holm_bonferroni = line.strip("\n").split("\t")

        if passes_holm_bonferroni in ["YES"]:
            output_file.write(f"{ngram}\n")