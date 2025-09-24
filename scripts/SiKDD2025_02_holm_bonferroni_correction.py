for file in ["chi_square_GENERAL_PRECEDING_ngrams.tsv",
             "chi_square_GENERAL_SUBSEQUENT_ngrams.tsv",
             "chi_square_CVC-ROBUST_PRECEDING_ngrams.tsv",
             "chi_square_CVC-ROBUST_SUBSEQUENT_ngrams.tsv",
             "chi_square_CVC-FINEGRAINED_PRECEDING_ngrams.tsv",
             "chi_square_CVC-FINEGRAINED_SUBSEQUENT_ngrams.tsv"]:

    lines_to_be_sorted = []

    read_file = open(file, "r", encoding="UTF-8").readlines()
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
        residual_interpretation = line.strip("\n").split("\t")

        lines_to_be_sorted.append([ngram, chi_square_statistic, p_value, robust_p_value, df, N, minimum_dimension, Cramers_V, max_residual, residual_interpretation])


    output_file = open(file.replace('.tsv', '_AFTER_HOLM-BONFERRONI.tsv'), "w", encoding="UTF-8")
    output_file.write("{}\n".format("\t".join([str(x) for x in ["ngram", "chi_square_statistic", "p_value", "robust_p_value", "df", "N", "minimum_dimension", "Cramers_V", "max_residual", "residual_interpretation", "alpha", "k", "m", "passes_holm_bonferroni"]])))

    # SORT LINES BY p_value
    sorted_lines = sorted(lines_to_be_sorted, key=lambda x: float(x[2]), reverse=False)

    # NUMBER OF p-values
    m = len(sorted_lines)

    # PARAMETER
    alpha = 0.05

    # SEQUENTIAL NUMBER FOR HYPOTHESES
    k = 0

    for sorted_line in sorted_lines:
        list_for_output = sorted_line
        k += 1  # INCREASE THE SEQUENTIAL NUMBER
        p_value_in_line = float(sorted_line[2])
        if p_value_in_line <= alpha/(m+1-k):
            passes_holm_bonferroni = "YES"
        else:
            passes_holm_bonferroni = "NO"
        list_for_output.append(alpha)
        list_for_output.append(k)
        list_for_output.append(m)
        list_for_output.append(passes_holm_bonferroni)


        output_file.write("{}\n".format("\t".join([str(x) for x in list_for_output])))

