from typing import Dict, List, Optional, Tuple, Union
from transformers import EvalPrediction



def strict_sentiment_f1(p: EvalPrediction) -> Dict:
    # https://zhuanlan.zhihu.com/p/363670628
    preds, labels = p
    print(preds)
    print(labels)
    


