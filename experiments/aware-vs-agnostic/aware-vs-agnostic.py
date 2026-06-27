from hinteval import Dataset
from hinteval.cores import Subset
from statistics import mean
from prettytable import PrettyTable
from draw_chart import draw_chart

def evaluate(subset: Subset):
    results = {'relevance': [], 'readability': [], 'convergence': [], 'familiarity': [], 'answer-leakage': []}
    instances = subset.get_instances()
    for instance in instances:
        for hint in instance.hints:
            relevance = hint.metrics['relevance-llm-meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo'].value
            readability = hint.metrics['readability-ml-xgboost-trf'].value
            convergence = hint.metrics['convergence-llm-llama-3-70b'].value
            familiarity = hint.metrics['familiarity-wikipedia-trf'].value
            answer_leakage = hint.metrics['answer-leakage-contextual-include_stop_words-trf'].value
            results['relevance'].append(relevance)
            results['readability'].append(readability)
            results['convergence'].append(convergence)
            results['familiarity'].append(familiarity)
            results['answer-leakage'].append(answer_leakage)
    for key in results:
        results[key] = round(mean(results[key]), 4)
    return results


def main():
    rows = []
    table = PrettyTable(
        ['Generator', 'Strategy', 'Relevance', 'Readability', 'Convergence', 'Familiarity', 'Answer Leakage'])
    dataset = Dataset.download_and_load_dataset('wikihint')
    generators = ['LLaMA-3.1-8b', 'LLaMA-3.1-70b', 'LLaMA-3.1-405b', 'GPT-4']
    for generator in generators:
        answer_aware = dataset[f'{generator}-Vanilla-answer-aware']
        aware_results = evaluate(answer_aware)
        aware_row = [generator, 'Aware', aware_results['relevance'], aware_results['readability'],
               aware_results['convergence'], aware_results['familiarity'], aware_results['answer-leakage']]
        rows.append(aware_row)

        answer_agnostic = dataset[f'{generator}-Vanilla-answer-agnostic']
        agnostic_results = evaluate(answer_agnostic)
        agnostic_row = [generator, 'Agnostic', agnostic_results['relevance'], agnostic_results['readability'],
               agnostic_results['convergence'], agnostic_results['familiarity'], agnostic_results['answer-leakage']]
        rows.append(agnostic_row)

    table.add_rows(rows)
    print(table)
    draw_chart(rows)

if __name__ == '__main__':
    main()
