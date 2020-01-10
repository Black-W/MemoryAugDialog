import collections
import math
import os
import random
import re
import sys
import codecs
from nltk import bigrams
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import numpy as np

''' FULL_BLEU
    1: Individual(Smooth & NonSmooth) + Cumulative(Smooth & NonSmooth)(NLTK & TF_NMT) 
    0: Cumulative(Smooth & NonSmooth)(TF_NMT)  [Default]
'''
FULL_BLEU = 1
DEBUG = 0

if 1 == FULL_BLEU:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def round_for_list(x, precision):
    return [round(data, precision) for data in x]


def get_individual_BLEU_1_to_4_by_nltk(references, candidate):
    scores = []
    if DEBUG >= 3:
        print("references:", references)
        print("candidate:", candidate)
    smoothing_function = SmoothingFunction().method0
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(1, 0, 0, 0), smoothing_function=smoothing_function))  # 1 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0, 1, 0, 0), smoothing_function=smoothing_function))  # 2 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0, 0, 1, 0), smoothing_function=smoothing_function))  # 3 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0, 0, 0, 1), smoothing_function=smoothing_function))  # 4 gram

    scores = round_for_list(scores, 4)
    print("Individual(Smooth):        \t" + "\t".join(map(str, scores)))
    return scores


def get_cumulative_BLEU_1_to_4_by_nltk(references, candidate):
    scores = []
    smoothing_function = SmoothingFunction().method0
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(1, 0, 0, 0), smoothing_function=smoothing_function))  # 1 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function))  # 2 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function))  # 3 gram
    scores.append(100 * corpus_bleu(references, candidate, \
                                    weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function))  # 4 gram
    scores = round_for_list(scores, 4)
    print("Cumulative(Smooth)(NLTK):\t" + "\t".join(map(str, scores)))
    return scores


def fetch_data(references_with_source_file, uniq_translation_file, is_merge_same_src=1):
    references_reader = codecs.open(references_with_source_file, 'r', 'utf-8')
    translation_reader = codecs.open(uniq_translation_file, 'r', 'utf-8')

    translations = []
    for line in translation_reader:
        slots = line.strip("\n").replace(' </s>', '').split()
        translations.append(slots)

    if 1 == is_merge_same_src:
        references = []
        last_query = ""
        refs_for_a_query = []
        for line in references_reader:
            slots = line.strip("\n").split("\t")
            query, query_emo, ref, ref_emo = slots
            # query, ref = slots
            if query != last_query:
                if len(refs_for_a_query) > 0:
                    references.append(refs_for_a_query)
                refs_for_a_query = []
                last_query = query
            ref = ref.split()
            refs_for_a_query.append(ref)
        references.append(refs_for_a_query)
    elif 0 == is_merge_same_src:
        references = []
        for line in references_reader:
            slots = line.strip("\n").split("\t")
            query, ref = slots
            ref = ref.split()
            refs_for_a_query = [ref]
            references.append(refs_for_a_query)

    if DEBUG >= 2:
        print("candidate size:", len(translations))
        print("references size:", len(references))
        for reference in references:
            print("reference len:", len(reference))
            print("reference:", reference)
    return references, translations


def get_bleus(references, translations):
    cumulative_smooth_set = []
    for N_of_gram in range(1, 5):  # BLEU-1 to BLEU-4
        bleu_score, precisions, bp, ratio, translation_length, reference_length \
            = compute_bleu(references, translations, max_order=N_of_gram, smooth=True)
        cumulative_smooth_set.append(100 * bleu_score)
    cumulative_smooth_set = round_for_list(cumulative_smooth_set, 4)

    cumulative_non_smooth_set = []
    for N_of_gram in range(1, 5):  # BLEU-1 to BLEU-4
        bleu_score, precisions, bp, ratio, translation_length, reference_length \
            = compute_bleu(references, translations, max_order=N_of_gram, smooth=False)
        cumulative_non_smooth_set.append(100 * bleu_score)
    cumulative_non_smooth_set = round_for_list(cumulative_non_smooth_set, 4)

    name_set = ["              Type             ", "Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"]
    print("\t".join(name_set))
    print("Cumulative(Smooth):        \t" + "\t".join(map(str, cumulative_smooth_set)))
    print("Cumulative(NonSmooth):        \t" + "\t".join(map(str, cumulative_non_smooth_set)))

    if 1 == FULL_BLEU:
        get_individual_BLEU_1_to_4_by_nltk(references, translations)
        get_cumulative_BLEU_1_to_4_by_nltk(references, translations)


def main(references_with_source_file, uniq_translation_file, is_merge_same_src=1):
    # 1: merge(for normal seq2seq predict) 0: not merge(for line by line predict)
    references, translations = fetch_data(references_with_source_file, uniq_translation_file, is_merge_same_src)
    get_bleus(references, translations)
    # print(f'nltk bleu-1:{corpus_bleu(references,translations,weights=(1,0,0,0))}')
    # print(f'nltk bleu-2:{corpus_bleu(references,translations,weights=(0.5,0.5,0,0))}')
    # print(f'nltk bleu-3:{corpus_bleu(references,translations,weights=(0.33,0.33,0.33,0))}')
    # print(f'nltk bleu-4:{corpus_bleu(references,translations,weights=(0.25,0.25,0.25,0.25))}')
    # 2: calculate dist1/dist2
    # 对于stc数据由于可能会有重复的问题，所以要得到独立的问题
    # test = open(os.path.join('data', 'weibo_utf8', 'test.txt'), 'r').readlines()
    # select = []
    # for i in range(len(test)):
    #     if i == len(test) - 1:
    #         select.append(True)
    #     elif test[i].split('\t')[0] == test[i + 1].split('\t')[0]:
    #         select.append(False)
    #     else:
    #         select.append(True)
    # select = np.array(select)
    # translations = np.array(translations)
    # translations = translations[select]
    total_unigrams = []
    total_bigrams = []
    for i in translations:
        # Calculate distinct-1 and distinct-2
        total_unigrams += i
        total_bigrams += bigrams(i)
    distinct_1 = len(set(total_unigrams)) / len(total_unigrams)
    distinct_2 = len(set(total_bigrams)) / len(total_bigrams)
    print("Distinct-1:\t{0}\tdistinct-2:\t{1}".format(distinct_1, distinct_2))


if __name__ == '__main__':

    references_with_source_file = os.path.join('../ECM/data', 'ESTC', 'test.txt')
    mark = 'stc'
    generate_mark = 'stc'

    # references_with_source_file = os.path.join('../ECM/data', 'NLPCC2017', 'valid.txt')
    # mark = 'nlpcc2017'
    # generate_mark = 'nlpcc2017'

    for i in [39]:
        modelID = str(i)
        # uniq_translation_file = os.path.join('result', f'res_gene_{mark}_tdv22_epoch{modelID}_{generate_mark}')
        uniq_translation_file = os.path.join('result', f'ESTC_best_no_emotion')
        # uniq_translation_file = os.path.join('..', 'ECM', 'result', 'nlpcc2017', 'result_epoch_300000.txt')
        print(mark, 'epoch: ', modelID)
        """
        references_with_source_file： 测试集，Post-response，一个post可以有多个不同的response，所以有多行的Post相同，response不同
        uniq_translation_file： 机器生成的回复，每行是一个结果，多行可能是相同的结果，对应同一个Post，这些相同的结果应该合并
        """
        main(references_with_source_file, uniq_translation_file, is_merge_same_src=1)
