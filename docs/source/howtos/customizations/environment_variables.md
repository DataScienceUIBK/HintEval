(how-to-guides-customizations-env-variables)=

# Environment Variables

The HintEval framework allows users to store checkpoints for evaluation metrics and hint generation tasks, making it
easy to resume processes if errors occur. Additionally, most evaluation methods require downloading and caching
pre-trained models to compute their corresponding metrics. These checkpoints and cached models are stored in specific
directories, which HintEval can manage based on user preferences.

### Caching Pre-trained Models

By default, pre-trained models are cached in the user's home directory under the following path:  
`~/.cache/hinteval`. If you want to change the cache directory to a location of your choice, you can modify the path
using the following code:

```python
import os

os.environ['HINTEVAL_CACHE_DIR'] = 'path/to/your/cache_directory'
```

Here’s the structure of the `cache_directory` directory:

```
📁 cache_directory
├── 📁 convergence-nn
│   ├── 📁 bert-base
│   └── 📁 roberta-large
├── 📁 convergence-specificity
│   ├── 📁 bert-base
│   └── 📁 roberta-large
├── 📁 embeddings
├── 📁 familiarity-freq
├── 📁 question-classification
├── 📁 readability-fe
├── 📁 readability-nn
│   ├── 📁 bert-base
│   └── 📁 roberta-large
├── 📁 relevance-contextual
│   ├── 📁 bert-base
│   └── 📁 roberta-large
└── 📁 relevance-non-contextual
```

### Checkpoint Storage

HintEval also supports checkpointing, allowing you to save progress during evaluations or hint generation. To customize
the directory where checkpoints are stored, use the following code to set the desired path:

```python
import os

os.environ['HINTEVAL_CHECKPOINT_DIR'] = 'path/to/your/checkpoint_directory'
```

Here’s the structure of the `checkpoint_directory` directory:

```
📁 checkpoint_directory
├── 📄 answer_leakage_contextual_exclude_stop_words.pickle
├── 📄 answer_leakage_contextual_include_stop_words.pickle
├── 📄 answer_leakage_lexical_exclude_stop_words.pickle
├── 📄 answer_leakage_lexical_include_stop_words.pickle
├── 📄 convergence_bert-base.pickle
├── 📄 convergence_llama-3-70b.pickle
├── 📄 convergence_roberta-large.pickle
├── 📄 familiarity_exclude_stop_words.pickle
├── 📄 familiarity_include_stop_words.pickle
├── 📄 familiarity_wikipedia.pickle
├── 📄 readability_automated_readability_index.pickle
├── 📄 readability_bert-base.pickle
├── 📄 readability_coleman_liau_index.pickle
├── 📄 readability_flesch_kincaid_reading_ease.pickle
├── 📄 readability_gunning_fog_index.pickle
├── 📄 readability_llm_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo.pickle
├── 📄 readability_random_forest.pickle
├── 📄 readability_roberta-large.pickle
├── 📄 readability_smog_index.pickle
├── 📄 readability_xgboost.pickle
├── 📄 relevance_contextual_bert-base.pickle
├── 📄 relevance_contextual_roberta-large.pickle
├── 📄 relevance_non_contextual_glove.42B.pickle
├── 📄 relevance_non_contextual_glove.6B.pickle
├── 📄 relevance_rouge1.pickle
├── 📄 relevance_rouge2.pickle
├── 📄 relevance_rougeL.pickle
├── 📄 specificity_bert-base.pickle
├── 📄 specificity_roberta-large.pickle
```

These environment variables give you control over where HintEval stores cache and checkpoint data, optimizing
performance based on your system setup.