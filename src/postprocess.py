import json
import re
import os
from pprint import pprint
from metrics import calc_sentiment_scores

pol_map = {
    "pos": "positive",
    "neg": "negative",
    "other": "neutral",
    "neu": "neutral",
}


def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


# A: (entity, aspect, opinion, sentiment)
# copied from `metrics/utils.py`, should keep consistent
def str_2_tuples(string: str) -> list[tuple]:
    """Convert a string of `A:(iphone, screen, small, negative), B:(iphone, screen, small, negative)` to a list of tuples of 4 elements
    When 4 elements are not present, fill with empty string; when more than 4 elements are present, trim to 4 elements
    """
    pattern = r"\((.*?)\)"
    tuples = re.findall(pattern, string)
    tuples = [tuple(t.split(",")) for t in tuples]
    for i, t in enumerate(tuples):
        if len(t) < 4:
            # manually add empty string until 4 or
            # trim the tuple to 4
            t += ("",) * (4 - len(t))
        elif len(t) > 4:
            t = t[:4]
        tuples[i] = t
    return tuples


def main(
    reference_path: str = "data/diaasq/jsons_en/valid.json",
    inference_path: str = "output/diaasq/jsons_en/valid.json",
    output_path: str = "output/diaasq/jsons_en/valid_formatted.json",
):
    """turn the output of the model into the format that can be used by the metrics, and save the formatted output

    Parameters
    ----------
    file_path : str
        a file path to the output of the model
        It should be in json format and each data should contain these keys `doc_id` and `output`
        `output`: str, output of the text2text model
        `doc_id`: str, id of each document matching those in diaasq
    """
    ref = load_json(reference_path)
    inf = load_json(inference_path)
    assert len(ref) == len(inf)
    # sort the data by doc_id
    ref = sorted(ref, key=lambda x: x["doc_id"])
    inf = sorted(inf, key=lambda x: x["doc_id"])

    # check if the doc_id matches
    for r, i in zip(ref, inf):
        assert r["doc_id"] == i["doc_id"]

    # ============================

    # turn gold to gold format
    # !! we turn the gold tripet to the form of (entity, aspect, opinion, sentiment)
    # !! and ignore the opinion holder

    for r in ref:
        raw_triplets = r["triplets"]
        processed_triplets = [
            (t[7], t[8], t[9], pol_map.get(t[6])) for t in raw_triplets
        ]
        r["triplets"] = processed_triplets

    sentiment_scores = calc_sentiment_scores(tokenizer=None)
    print(" SCORES ".center(80, "="))
    pprint(sentiment_scores(ref, inf))

    # directly metrics on the output of the model
    for ref_data, inf_data in zip(ref, inf):
        inf_data["triplets"] = str_2_tuples(inf_data["output"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(inf, output_path)
    print(f"Save the formatted output to {output_path}")


if __name__ == "__main__":
    main()
