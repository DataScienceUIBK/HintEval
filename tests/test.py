import os
from hinteval import Dataset
from hinteval.evaluation.relevance import Rouge, NonContextualEmbeddings, ContextualEmbeddings as RelCon, LlmBased as RelLLM
from hinteval.evaluation.readability import TraditionalIndexes, MachineLearningBased, NeuralNetworkBased as RedNN, LlmBased as RedLLM
from hinteval.evaluation.convergence import Specificity, NeuralNetworkBased as ConvNN, LlmBased as ConvLLM
from hinteval.evaluation.familiarity import WordFrequency, Wikipedia
from hinteval.evaluation.answer_leakage import Lexical, ContextualEmbeddings as ALCon
from hinteval.cores import Subset, Instance
from hinteval.model import AnswerAware

api_key = 'your_api_key'
model_name_remote = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
model_name_local = 'meta-llama/Meta-Llama-3-8B-Instruct'

os.environ["HINTEVAL_CHECKPOINT_DIR"] = "./checkpoints"
os.environ["HINTEVAL_CACHE_DIR"] = "./cache"

answer_aware = AnswerAware(model_name=model_name_remote, api_key=api_key, num_of_hints=5, parse_llm_response=None,
                           temperature=0.7, top_p=1.0, max_tokens=512, batch_size=2,
                           checkpoint=True, checkpoint_step=3, enable_tqdm=True)

Dataset.available_datasets(True, True)
dataset = Dataset.download_and_load_dataset('WikiHint')
Dataset.download_and_load_dataset('TriviaHG')
Dataset.download_and_load_dataset('HintQA')
Dataset.download_and_load_dataset('kg-hint')
instances = dataset['train'].get_instances()[:30]

entire = Subset()
for instance in instances:
    question = instance.question.question
    answers = [answer.answer for answer in instance.answers]
    entire.add_instance(Instance.from_strings(question, answers, []))

answer_aware.generate(entire.get_instances())

dataset = Dataset('new_dataset')
dataset.add_subset(entire)

dataset.prepare_dataset(enable_tqdm=True)
instances = dataset['entire'].get_instances()

"""""""""""Relevance"""""""""""
"""""""""""Relevance - Rouge"""""""""""
relevance_rough = Rouge('rouge1', enable_tqdm=True)
res_1 = relevance_rough.evaluate(instances)
relevance_rough.release_memory()

relevance_rough = Rouge('rouge2', enable_tqdm=True)
res_2 = relevance_rough.evaluate(instances)
relevance_rough.release_memory()

relevance_rough = Rouge('rougeL', checkpoint=True, checkpoint_step=10, enable_tqdm=True)
res_3 = relevance_rough.evaluate(instances)
relevance_rough.release_memory()

"""""""""""Relevance - Non Contextual"""""""""""
relevance_non_contextual = NonContextualEmbeddings('glove.6B', enable_tqdm=True)
relevance_non_contextual.evaluate(instances)
relevance_non_contextual.release_memory()

relevance_non_contextual = NonContextualEmbeddings('glove.42B', enable_tqdm=True)
relevance_non_contextual.evaluate(instances)
relevance_non_contextual.release_memory()

"""""""""""Relevance - Contextual"""""""""""
relevance_contextual = RelCon('bert-base', checkpoint=True, checkpoint_step=5, enable_tqdm=True)
relevance_contextual.evaluate(instances)
relevance_contextual.release_memory()

relevance_contextual = RelCon('roberta-large', checkpoint=True, checkpoint_step=10, enable_tqdm=True)
relevance_contextual.evaluate(instances)
relevance_contextual.release_memory()

"""""""""""Relevance - LLM"""""""""""
relevance_llm = RelLLM(model_name=model_name_remote.replace('8', '70'), api_key=api_key, checkpoint=True,
                              checkpoint_step=5, enable_tqdm=True)
relevance_llm.evaluate(instances)
relevance_llm.release_memory()

relevance_llm = RelLLM(model_name=model_name_local, api_key=None, checkpoint=True, checkpoint_step=5,
                              enable_tqdm=True)
relevance_llm.evaluate(instances)
relevance_llm.release_memory()

"""""""""""Relevance"""""""""""

"""""""""""Readability"""""""""""
questions = [instance.question for instance in instances]
hints = []
[hints.extend(instance.hints) for instance in instances]

"""""""""""Readability - Traditional"""""""""""
readability_traditional = TraditionalIndexes('flesch_kincaid_reading_ease', spacy_pipeline='en_core_web_sm',
                                             checkpoint=True, checkpoint_step=50, enable_tqdm=True)
readability_traditional.evaluate(hints + questions)
readability_traditional.release_memory()

readability_traditional = TraditionalIndexes('gunning_fog_index', spacy_pipeline='en_core_web_sm', enable_tqdm=True)
readability_traditional.evaluate(hints + questions)
readability_traditional.release_memory()

readability_traditional = TraditionalIndexes('smog_index', spacy_pipeline='en_core_web_sm', enable_tqdm=True)
readability_traditional.evaluate(hints + questions)
readability_traditional.release_memory()

readability_traditional = TraditionalIndexes('coleman_liau_index', spacy_pipeline='en_core_web_sm', checkpoint=True,
                                             checkpoint_step=50, enable_tqdm=True)
readability_traditional.evaluate(hints + questions)
readability_traditional.release_memory()

readability_traditional = TraditionalIndexes('automated_readability_index', spacy_pipeline='en_core_web_sm',
                                             enable_tqdm=True)
readability_traditional.evaluate(hints + questions)
readability_traditional.release_memory()

"""""""""""Readability - Machine Learning"""""""""""
readability_ml = MachineLearningBased('xgboost', spacy_pipeline='en_core_web_sm', checkpoint=True, checkpoint_step=100,
                                    enable_tqdm=True)
readability_ml.evaluate(hints + questions)
readability_ml.release_memory()

readability_ml = MachineLearningBased('random_forest', force_download=True, checkpoint=True, checkpoint_step=100,
                                    spacy_pipeline='en_core_web_lg', enable_tqdm=True)
readability_ml.evaluate(hints + questions)
readability_ml.release_memory()

"""""""""""Readability - Neural Network"""""""""""
readability_nn = RedNN('bert-base', checkpoint=True, checkpoint_step=2, enable_tqdm=True)
readability_nn.evaluate(hints + questions)
readability_nn.release_memory()

readability_nn = RedNN('roberta-large', checkpoint=True, checkpoint_step=2, enable_tqdm=True)
readability_nn.evaluate(hints + questions)
readability_nn.release_memory()

"""""""""""Readability - LLM"""""""""""
relevance_contextual = RedLLM(model_name=model_name_remote, api_key=api_key, batch_size=2, checkpoint=False,
                              checkpoint_step=5, enable_tqdm=True)
relevance_contextual.evaluate(hints + questions)
relevance_contextual.release_memory()

relevance_contextual = RedLLM(model_name=model_name_local, api_key=None, batch_size=1, checkpoint=True,
                              checkpoint_step=5,
                              enable_tqdm=True, max_tokens=128)
relevance_contextual.evaluate(hints + questions)
relevance_contextual.release_memory()

"""""""""""Readability"""""""""""

"""""""""""Convergence"""""""""""
"""""""""""Convergence - Specificity"""""""""""
convergence_specificity = Specificity('bert-base', checkpoint=True, checkpoint_step=5, enable_tqdm=True)
convergence_specificity.evaluate(instances)
convergence_specificity.release_memory()

convergence_specificity = Specificity('roberta-large', checkpoint=True, checkpoint_step=5, enable_tqdm=True)
convergence_specificity.evaluate(instances)
convergence_specificity.release_memory()

"""""""""""Convergence - NeuralNetwork"""""""""""
convergence_nn = ConvNN('bert-base', checkpoint=True, checkpoint_step=5, enable_tqdm=True)
convergence_nn.evaluate(instances)
convergence_nn.release_memory()

convergence_nn = ConvNN('roberta-large', checkpoint=True, checkpoint_step=5, enable_tqdm=True)
convergence_nn.evaluate(instances)
convergence_nn.release_memory()

"""""""""""Convergence - LLM"""""""""""
convergence_llm = ConvLLM('llama-3-8b', together_ai_api_key=None, checkpoint=True, checkpoint_step=5,
                          enable_tqdm=True)
convergence_llm.evaluate(instances)
convergence_llm.release_memory()

convergence_llm = ConvLLM('llama-3-70b', together_ai_api_key=api_key, checkpoint=True, checkpoint_step=5,
                          enable_tqdm=True)
convergence_llm.evaluate(instances)
convergence_llm.release_memory()

"""""""""""Convergence"""""""""""

"""""""""""Familiarity"""""""""""
questions = [instance.question for instance in instances]
answers = [instance.answers[0] for instance in instances]
hints = []
[hints.extend(instance.hints) for instance in instances]

"""""""""""Familiarity - Word Frequency"""""""""""
familiarity_llm = WordFrequency('include_stop_words', spacy_pipeline='en_core_web_sm', checkpoint=True,
                                checkpoint_step=5, enable_tqdm=True)
familiarity_llm.evaluate(questions + hints + answers)
familiarity_llm.release_memory()

familiarity_llm = WordFrequency('exclude_stop_words', force_download=True, spacy_pipeline='en_core_web_sm',
                                enable_tqdm=True)
familiarity_llm.evaluate(questions + hints + answers)
familiarity_llm.release_memory()

"""""""""""Familiarity - Wikipedia"""""""""""

familiarity_llm = Wikipedia(spacy_pipeline='en_core_web_sm', checkpoint=True, checkpoint_step=30, enable_tqdm=True)
familiarity_llm.evaluate(questions + hints + answers)
familiarity_llm.release_memory()

"""""""""""Familiarity"""""""""""

"""""""""""AnswerLeakage"""""""""""
"""""""""""AnswerLeakage - Lexical"""""""""""
answer_leakage_lexical = Lexical('include_stop_words', spacy_pipeline='en_core_web_sm', checkpoint=True,
                                 checkpoint_step=10, enable_tqdm=True)
answer_leakage_lexical.evaluate(instances)
answer_leakage_lexical.release_memory()

answer_leakage_lexical = Lexical('exclude_stop_words', spacy_pipeline='en_core_web_sm', checkpoint=True,
                                 checkpoint_step=10, enable_tqdm=True)
answer_leakage_lexical.evaluate(instances)
answer_leakage_lexical.release_memory()

"""""""""""AnswerLeakage - Contextual"""""""""""
answer_leakage_contextual = ALCon('all-mpnet-base-v2', 'include_stop_words', spacy_pipeline='en_core_web_sm',
                                  checkpoint=True, checkpoint_step=10, enable_tqdm=True)
answer_leakage_contextual.evaluate(instances)
answer_leakage_contextual.release_memory()

answer_leakage_contextual = ALCon('all-mpnet-base-v2', 'exclude_stop_words', spacy_pipeline='en_core_web_md',
                                  checkpoint=True, checkpoint_step=10, enable_tqdm=True)
answer_leakage_contextual.evaluate(instances)
answer_leakage_contextual.release_memory()

"""""""""""AnswerLeakage"""""""""""

# print(instances[:2])
dataset.store_json('./new_dataset.json')
