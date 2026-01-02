from hinteval import Dataset
from hinteval.cores import Subset
from scipy.stats import spearmanr, kendalltau
from draw_chart import draw_chart
from prettytable import PrettyTable


# from draw_chart import draw_chart

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
    results = {'relevance': [], 'readability': [], 'convergence': [], 'familiarity': [], 'answer-leakage': []}
    instances = subset.get_instances()
    for instance in instances:
        for hint in instance.hints:
            relevance = hint.metrics['relevance-llm-meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo'].value
            readability = hint.metrics['readability-llm-meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo'].value
            convergence = hint.metrics['convergence-llm-llama-3-70b'].value
            familiarity = hint.metrics['familiarity-wikipedia-trf'].value
            answer_leakage = hint.metrics['answer-leakage-contextual-include_stop_words-trf'].value
            results['relevance'].append(relevance)
            results['readability'].append(readability)
            results['convergence'].append(convergence)
            results['familiarity'].append(familiarity)
            results['answer-leakage'].append(answer_leakage)
    return results


def main():
    datasets = ['triviahg', 'wikihint']
    correlations = {'Spearmans': spearmanr, 'Kendalltau': kendalltau}

    full_results = dict()
    for dataset_name in datasets:
        if dataset_name not in full_results.keys():
            full_results[dataset_name] = dict()
        dataset = Dataset.download_and_load_dataset(dataset_name)
        subset = dataset['test']
        results = evaluate(subset)
        for corr_name in correlations:
            if corr_name not in full_results[dataset_name].keys():
                full_results[dataset_name][corr_name] = []
            for met_1 in results.keys():
                row = [met_1]
                for met_2 in results.keys():
                    corr = round(correlations[corr_name](results[met_1], results[met_2])[0], 4)
                    row.append(corr)
                full_results[dataset_name][corr_name].append(row)
    draw_chart(full_results)

if __name__ == '__main__':
    main()
