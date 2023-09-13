from typing import List, Dict
from collections import defaultdict
from copy import deepcopy
import pickle
import json


def load_pkl(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: list[dict], path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def remove_space(example: dict) -> dict:
    _example = deepcopy(example)
    sentences = _example["sentences"]
    for i, sent in enumerate(sentences):
        sentences[i] = _remove_space(sent)
    return _example


def _remove_space(text):
    # such that the texts look more natural
    text = text.split()
    text = "".join(text)
    return text


def char_to_number(char: str):
    assert char < "Z", "char must be in [A, Z]"
    return ord(char) - ord("A")


def number_to_char(number: int):
    assert (
        number < 26
    ), "number must be in [0, 25] such that it can be converted to a char"
    return chr(number + ord("A"))


def get_metadata(data: Dict[str, any]) -> Dict[str, any]:
    # number of aspects, repeated aspects
    # number of entities, repeated entities
    # number of entity-aspect pairs, repeated entity-aspect pairs
    # aspect count
    # entity count
    # entity-aspect count

    aspect_count_dict = defaultdict(int)
    entity_count_dict = defaultdict(int)
    entity_aspect_count_dict = defaultdict(int)
    gold_aspects = [a[-1] for a in data["aspects"]]
    gold_entities = [e[-1] for e in data["targets"]]

    gold_ent_aspects = []
    for tup in data["triplets"]:
        gold_ent_aspects.append((tup[7], tup[8]))

    for aspect in gold_aspects:
        if aspect.strip() == "":
            continue
        aspect_count_dict[aspect] += 1
    for entity in gold_entities:
        if entity.strip() == "":
            continue
        entity_count_dict[entity] += 1

    for ent, asp in gold_ent_aspects:
        if ent.strip() == "" or asp.strip() == "":
            continue
        entity_aspect_count_dict[(ent, asp)] += 1

    # =================================================
    # pos, neg, neu count for the full dialogue, regardless of entity/aspect
    # pos, neg, neu count for each entity
    # pos, neg, neu count for each aspect
    entity_senti_number_dict = defaultdict(lambda: defaultdict(int))
    aspect_senti_number_dict = defaultdict(lambda: defaultdict(int))
    for tup in data["triplets"]:
        entity = tup[7]
        senti = tup[6]
        if entity.strip() == "":
            continue
        entity_senti_number_dict[entity][senti] += 1

        aspect = tup[8]
        if aspect.strip() == "":
            continue
        aspect_senti_number_dict[aspect][senti] += 1
    repeated_entity_aspect_count = sum(
        [v for v in entity_aspect_count_dict.values() if v > 1]
    )
    if repeated_entity_aspect_count > 0:
        entity_aspect_senti_number_dict = defaultdict(lambda: defaultdict(int))
        for tup in data["triplets"]:
            entity = tup[7]
            aspect = tup[8]
            senti = tup[6]
            if aspect.strip() == "" or entity.strip() == "":
                continue
            # use f"{entity}_{aspect}"
            # to avoid TypeError: keys must be str, int, float, bool or None, not tuple when saving to json
            entity_aspect_senti_number_dict[f"{entity}_{aspect}"][senti] += 1

    # =================================================
    metadata = {
        "aspect_count": len(aspect_count_dict),
        "entity_count": len(entity_count_dict),
        "entity_aspect_count": len(entity_aspect_count_dict),
        "repeated_aspect_count": sum([v for v in aspect_count_dict.values() if v > 1]),
        "repeated_entity_count": sum([v for v in entity_count_dict.values() if v > 1]),
        "repeated_entity_aspect_count": repeated_entity_aspect_count,
        "entity_senti_number_dict": entity_senti_number_dict,
        "aspect_senti_number_dict": aspect_senti_number_dict,
    }
    if repeated_entity_aspect_count > 0:
        metadata["entity_aspect_senti_number_dict"] = entity_aspect_senti_number_dict
    return metadata
