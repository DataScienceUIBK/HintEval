from hinteval import Dataset
from hinteval.cores import Subset
from statistics import mean
from prettytable import PrettyTable
from draw_chart import draw_chart

# 'relevance-contextual-roberta-large',
# 'relevance-rouge1',
# 'relevance-rouge2',
# 'relevance-rougeL',
# 'relevance-non-contextual-6B-trf',
# 'relevance-non-contextual-42B-trf',
# 'relevance-contextual-bert-base',
# 'relevance-llm-meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo',
#
# 'readability-flesch_kincaid_reading_ease-trf',
# 'readability-gunning_fog_index-trf',
# 'readability-smog_index-trf',
# 'readability-coleman_liau_index-trf',
# 'readability-automated_readability_index-trf',
# 'readability-ml-xgboost-trf',
# 'readability-ml-random_forest-trf',
# 'readability-nn-bert-base',
# 'readability-nn-roberta-large',
# 'readability-llm-meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo',
#
# 'convergence-specificity-bert-base',
# 'convergence-specificity-roberta-large',
# 'convergence-nn-bert-base',
# 'convergence-nn-roberta-large',
# 'convergence-llm-llama-3-70b'

# 'familiarity-freq-include_stop_words-trf',
# 'familiarity-freq-exclude_stop_words-trf',
# 'familiarity-wikipedia-trf',

# 'answer-leakage-lexical-include_stop_words-trf',
# 'answer-leakage-lexical-exclude_stop_words-trf',
# 'answer-leakage-contextual-include_stop_words-trf',
# 'answer-leakage-contextual-exclude_stop_words-trf',



def evaluate(subset: Subset):
    results = {'convergence': [], 'answer-leakage': []}
    instances = subset.get_instances()
    for instance in instances:
        for hint in instance.hints:
            convergence = hint.metrics['convergence-llm-llama-3-70b'].value
            answer_leakage = hint.metrics['answer-leakage-contextual-exclude_stop_words-trf'].value
            if convergence == 0:
                continue
            results['convergence'].append(convergence)
            results['answer-leakage'].append(answer_leakage)

    return results


def main():
    dataset = Dataset.download_and_load_dataset('wikihint')
    strategies = ['-Vanilla-answer-agnostic', '-Vanilla-answer-aware']
    print(dataset.get_subsets_name())
    generators = ['GPT-4']

    rows = dict()
    for strategy in strategies:
        if strategy not in rows:
            rows[strategy] = {'convergence': [], 'answer-leakage': []}
        for generator in generators:
            subset = dataset[f'{generator}{strategy}']
            results = evaluate(subset)
            for key in results.keys():
                rows[strategy][key].extend(results[key])
    # print(rows)
    draw_chart(rows)

if __name__ == '__main__':
    main()
