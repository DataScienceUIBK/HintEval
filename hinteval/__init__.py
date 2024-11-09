import os
import warnings

if 'HINTEVAL_CACHE_DIR' not in os.environ:
    os.environ['HINTEVAL_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.cache', 'hinteval')
if 'HINTEVAL_CHECKPOINT_DIR' not in os.environ:
    os.environ['HINTEVAL_CHECKPOINT_DIR'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
warnings.filterwarnings("ignore")

import nest_asyncio
nest_asyncio.apply()

from hinteval.cores.dataset.dataset import Dataset