import unittest
from dataset.diaasq.base import BaseDiaAsqDataset
import gc
from pathlib import Path

class TestBaseDiaAsqDataset(unittest.TestCase):
    def setUp(self):
        self.args = {
            'src': 'en',
            'train_split_name': 'train',
            'test_split_name': 'valid',
            'data_root': 'data/diaasq/dataset'
        }
        self.base_diaasq = BaseDiaAsqDataset(**self.args)
    def tearDown(self):
        self.args = None
        self.base_diaasq = None
        del self.args
        del self.base_diaasq
        gc.collect() # Ask YiTing 這樣實際到底做了什麼

    def test_determine_path(self):
        base_diaasq_trainpath = str(self.base_diaasq.train_file_path)
        expected_trainpath = str(Path(f'{self.args["data_root"]}/jsons_{self.args["src"]}/{self.args["train_split_name"]}.json'))

        self.assertEqual(base_diaasq_trainpath, expected_trainpath)

    def test_data_example(self):
        self.base_diaasq.read_json(self.base_diaasq.train_file_path)
        data =  {
        "doc_id": "0002",
        "sentences": [
            "This phone is not very good , but compared to the iPhone , I think it is better than the iPhone except for the processor [ laughs cry ]",
            "The iPhone is excellent as the processor and iOS , and others have been beaten by Android for many years .",
            "really . Sales also beat Android . Android manufacturers claim to be high - end and high - end every day , but they are just children in front of Apple .",
            "Samsung , Xiaomi does not all exceed Apple?Because there are too many Android systems , there is only one iOS . If there is only one Android , what do you think of the result ?",
            "As you say , I have n't used Xiaomi , so I ca n't comment . But traveling , my friend 's Xiaomi phone never took good photos . Especially when went to Malinghe Waterfall this time , we had to take pictures . Every photo taken by my brother 's Mi 11 was blurry . This experience is also speechless .",
            "Xiaomi 11 is really not good [ black line ] [ black line ] [ black line ] .",
            "The parameters overwhelme every year , and the experience is general every year ... that 's all . The phone is yours , who uses it , who knows .",
            "Even the experience is better than the iPhone . I ca n't stand the backward image system of the iPhone ."
        ],
        "replies": [
            -1,
            0,
            1,
            2,
            0,
            4,
            0,
            6
        ],
        "speakers": [
            0,
            1,
            2,
            1,
            3,
            0,
            3,
            0
        ],
        "triplets": [
            [
                20,
                21,
                24,
                25,
                17,
                18,
                "pos",
                "iPhone",
                "processor",
                "better"
            ],
            [
                30,
                31,
                35,
                36,
                32,
                33,
                "pos",
                "iPhone",
                "processor",
                "excellent"
            ],
            [
                30,
                31,
                37,
                38,
                32,
                33,
                "pos",
                "iPhone",
                "iOS",
                "excellent"
            ],
            [
                30,
                31,
                52,
                53,
                54,
                55,
                "pos",
                "iPhone",
                "Sales",
                "beat"
            ],
            [
                82,
                83,
                52,
                53,
                88,
                89,
                "pos",
                "Samsung",
                "Sales",
                "exceed"
            ],
            [
                84,
                85,
                52,
                53,
                88,
                89,
                "pos",
                "Xiaomi",
                "Sales",
                "exceed"
            ],
            [
                180,
                182,
                145,
                146,
                184,
                186,
                "neg",
                "Xiaomi 11",
                "photos",
                "not good"
            ],
            [
                248,
                249,
                244,
                246,
                243,
                244,
                "neg",
                "iPhone",
                "image system",
                "backward"
            ],
            [
                236,
                237,
                231,
                232,
                233,
                234,
                "neg",
                "iPhone",
                "experience",
                "better"
            ],
            [
                169,
                171,
                145,
                146,
                172,
                173,
                "neg",
                "Mi 11",
                "photos",
                "blurry"
            ],
            [
                169,
                171,
                175,
                176,
                178,
                179,
                "neg",
                "Mi 11",
                "experience",
                "speechless"
            ]
        ],
        "targets": [
            [
                20,
                21,
                "iPhone"
            ],
            [
                30,
                31,
                "iPhone"
            ],
            [
                82,
                83,
                "Samsung"
            ],
            [
                84,
                85,
                "Xiaomi"
            ],
            [
                169,
                171,
                "Mi 11"
            ],
            [
                180,
                182,
                "Xiaomi 11"
            ],
            [
                236,
                237,
                "iPhone"
            ],
            [
                248,
                249,
                "iPhone"
            ]
        ],
        "aspects": [
            [
                207,
                208,
                "experience"
            ],
            [
                199,
                201,
                "The parameters"
            ],
            [
                52,
                53,
                "Sales"
            ],
            [
                231,
                232,
                "experience"
            ],
            [
                244,
                246,
                "image system"
            ],
            [
                37,
                38,
                "iOS"
            ],
            [
                35,
                36,
                "processor"
            ],
            [
                24,
                25,
                "processor"
            ],
            [
                145,
                146,
                "photos"
            ],
            [
                175,
                176,
                "experience"
            ]
        ],
        "opinions": [
            [
                17,
                18,
                "better",
                "pos"
            ],
            [
                209,
                210,
                "general",
                "neg"
            ],
            [
                172,
                173,
                "blurry",
                "neg"
            ],
            [
                184,
                186,
                "not good",
                "neg"
            ],
            [
                54,
                55,
                "beat",
                "pos"
            ],
            [
                178,
                179,
                "speechless",
                "neg"
            ],
            [
                201,
                202,
                "overwhelme",
                "pos"
            ],
            [
                233,
                234,
                "better",
                "neg"
            ],
            [
                32,
                33,
                "excellent",
                "pos"
            ],
            [
                243,
                244,
                "backward",
                "neg"
            ],
            [
                88,
                89,
                "exceed",
                "pos"
            ]
        ]
        }

        self.assertEqual(self.base_diaasq.data[0], data)