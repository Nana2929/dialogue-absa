from collections import defaultdict, namedtuple

import pandas as pd

"""
Goal: calculate eval on ARTS when manipulating the sentiment order in the triplets.
"""

predict_path = (
    "output/llm/output_llama2/laptop_test_enriched_converted_usediff-True.jsonl"
)
gt_path = "data/arts/laptop_test_enriched_converted_usediff-True.json"


predict = pd.read_json(predict_path, orient="records", lines=True)
answer = pd.read_json(gt_path)


pred_tuple = namedtuple("pred_tuple", ["aspect", "sentiment"])
ans_tuple = namedtuple(
    "ans_tuple", ["aspect", "first_sentiment", "most_sentiment", "senti_dict"]
)


# ****************
#  postprocessing
# ****************
predictions = []
answers = []
for idx, row in predict.iterrows():
    tmp = []
    s = row["generated_text"].split(", ")
    aspect = s[0].lstrip(" (").lower()
    senti = s[1].rstrip(")")

    pred_t = pred_tuple(aspect, senti)
    predictions.append(pred_t)

for idx, row in answer.iterrows():
    s = row["output"].split("A: ")
    aspect = s[1].split(", ")[0].lstrip("(").lower()
    ans_t = ans_tuple(
        aspect, row["first_sentiment"], row["most_sentiment"], row["senti_number"]
    )
    answers.append(ans_t)

# ****************
#  evaluation
# ****************

abbrev_map = {"positive": "pos", "negative": "neg", "neutral": "neu"}


import numpy as np

datas = {"align_first": [], "align_most": [], "most==first": []}
stats = {"align_first": 0, "align_most": 0, "most==first": 0, "grounded": 0}
for idx, (p, a) in enumerate(zip(predictions, answers)):
    if p.aspect == a.aspect:
        if p.sentiment == a.first_sentiment or p.sentiment == a.most_sentiment:
            stats["grounded"] += 1
        if p.sentiment == a.first_sentiment:
            stats["align_first"] += 1
            datas["align_first"].append((p, a))
        if p.sentiment == a.most_sentiment:
            stats["align_most"] += 1
            datas["align_most"].append((p, a))
        if p.sentiment == a.first_sentiment == a.most_sentiment:
            stats["most==first"] += 1


print(f"total: {len(answers)}")
print(f'#correct-aspect and sentiment appears in GT: {stats["grounded"]}')
print(f'#align_first: {stats["align_first"]}')
print(f'#align_most: {stats["align_most"]}')
print(f'#most==first: {stats["most==first"]}')
assert (
    stats["align_first"] + stats["align_most"] - stats["most==first"]
    == stats["grounded"]
)
# ****************
# check if a.senti_number has most_sentiment higher, is the probability of most-sent voting higher?
# ****************
import numpy as np
from scipy import stats

# H0: model selects most_sentiment instead of first_sentiment, does not depend on the number of most_sentiment
# H1: model selects most_sentiment instead of first_sentiment, depends on the number of most_sentiment (the more the better)

is_align_most_array = []
most_senti_number_array = []

for p, a in zip(predictions, answers):
    if p.aspect == a.aspect:
        if p.sentiment == a.first_sentiment or p.sentiment == a.most_sentiment:
            # correct
            if p.sentiment == a.first_sentiment == a.most_sentiment:
                # most sentiment = first sentiment, remove
                continue
            is_align_most_array.append(p.sentiment == a.most_sentiment)
            most_senti_number_array.append(a.senti_dict[abbrev_map[a.most_sentiment]])
assert len(is_align_most_array) == len(most_senti_number_array)
print(is_align_most_array)
print(most_senti_number_array)
print("#of data points: ", len(is_align_most_array))
# Pearson correlation
pearson_corr, pearson_p_value = stats.pearsonr(
    is_align_most_array, most_senti_number_array
)
print(f"Pearson Correlation Coefficient: {pearson_corr}")
print(f"Pearson Correlation P-value: {pearson_p_value}")

# Interpret the Pearson correlation
if pearson_p_value < 0.05:
    print("There is a statistically significant Pearson correlation.")
else:
    print("There is no statistically significant Pearson correlation.")
