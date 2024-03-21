import json

import fire
import pandas as pd

pol_map = {
    'pos': 'positive',
    'neg': 'negative',
    'neu': 'neutral'
}

def number_to_char(number: int):
    # 0 => A, 1 => B, ...
    return chr(number + 65)

def main(data_path: str = "data/arts/laptop_test_enriched.json",
         output_path: str = None,
         use_different_person: bool = True) :

    data = pd.read_json(data_path, orient= "index")
    data_idx = data.groupby('id')
    output = []
    id = 0

    for g in data_idx.groups:
        group = data_idx.get_group(g)

        d = dict()
        text_answer_list = []
        senti_number = {'pos': 0, 'neg': 0, 'neu': 0}
        person_id = 0
        for i, row in group.iterrows():
            text = f"{row['sentence']}"
            answer = f"({row['term']}, {row['polarity']})"
            text_answer_list.append((text, answer, row['polarity']))

            if row['polarity'] == 'positive':
                senti_number['pos'] += 1
            elif row['polarity'] == 'negative':
                senti_number['neg'] += 1
            elif row['polarity'] == 'neutral':
                senti_number['neu'] += 1
            person_id += 1
        most_sentiment = max(senti_number, key=senti_number.get)
        most_sentiment_complete_name = pol_map[most_sentiment]
        # try to order a text that != most_sentiment to the beginning
        text_answer_list.sort(key=lambda x: x[2] != most_sentiment_complete_name, reverse=True)
        # get first pol
        first_pol = text_answer_list[0][2]

        # add speaker
        if use_different_person:
            for i, (text, answer, pol) in enumerate(text_answer_list):
                text_answer_list[i] = (f"{number_to_char(i)}: {text}", f"{number_to_char(i)}: {answer}", pol)
        else:
            for i, (text, answer, pol) in enumerate(text_answer_list):
                text_answer_list[i] = (f"{number_to_char(0)}: {text}", f"{number_to_char(0)}: {answer}", pol)


        d['index'] = id
        d['instruction'] = 'Given a dialogue, please extract sentiment pairs in the form (aspect, sentiment polarity).'
        d['input'] = " ".join([x[0] for x in text_answer_list]).strip(',').strip()
        d['output'] = " ".join([x[1] for x in text_answer_list]).strip(',').strip()
        d['senti_number'] = senti_number
        d['most_sentiment'] = most_sentiment_complete_name
        d['first_sentiment'] = first_pol

        output.append(d)
        id += 1
    if output_path is None:
        output_path = data_path.replace(".json", f"_converted_usediff-{use_different_person}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Saving {len(output)} data to {output_path}")

if __name__ == '__main__':
    fire.Fire(main)