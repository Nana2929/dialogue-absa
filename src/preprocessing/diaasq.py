from pprint import pprint
from dataset.utils import (
    save_json,
)
from dataset.diaasq.full_dialogue_dataset import FullDiaAsqDataset

train_path = "data/diaasq/dataset/jsons_en/train.json"
test_path = "data/diaasq/dataset/jsons_en/valid.json"
instruct_path = "prompt/experiment/diaasq-fulldialog-en-llama-lora"
trainset = FullDiaAsqDataset("en", train_path, instruct_path)
testset = FullDiaAsqDataset("en", test_path, instruct_path)
# for i in range(10):
#     sample = dataset[i]
#     print(sample["input"])
#     print(sample["output"])
#     pprint(sample["metadata"])
#     print("====================================")

converted_train = list(trainset)
converted_test = list(testset)
print("train: ", len(converted_train))
print("valid: ", len(converted_test))
save_json(converted_train, "data/diaasq/dataset/jsons_en/diaasq_train_converted.json")
save_json(converted_test, "data/diaasq/dataset/jsons_en/diaasq_valid_converted.json")
