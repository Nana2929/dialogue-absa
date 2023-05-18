import unittest
from dataset.diaasq.base import BaseDiaAsq
import gc
from pathlib import Path

class TestBaseDiaAsq(unittest.TestCase):
    def setUp(self):
        self.args = {
            'src': 'en',
            'split': 'train',
            'data_root': 'data/diaasq/dataset'
        }
        self.base_diaasq = BaseDiaAsq(**self.args)
    def tearDown(self):
        self.args = None
        self.base_diaasq = None
        del self.args
        del self.base_diaasq
        gc.collect() # Ask YiTing 這樣實際到底做了什麼

    def test_determine_path(self):
        # string-lize
        base_diaasq_filepath = str(self.base_diaasq.filepath)
        expected_filepath = str(Path(f'{self.args["data_root"]}/jsons_{self.args["src"]}/{self.args["split"]}.json'))

        self.assertEqual(base_diaasq_filepath, expected_filepath)

    def test_data_example(self):
        self.base_diaasq.read_json()
        data = {
        "doc_id": "0003",
        "sentences": [
            "Really ? why I use IQOO9 for so many days that I feel very fragrant .",
            "If you really listen to digital bloggers , you can only buy that kind of cost - effective machine , but does everyone need it ? Does everyone play Yuanshen ?",
            "Ah , is there still a cost -effective machine for the recent machine ? Is not all the hardware shrinks more than last year , the price rises ? [ laugh haha ] .",
            "The sub -brand is basically cost - effective ( the pile last year ) .",
            "9 with DC is definitely fragrant but you try again in summer .",
            "In summer , 865 , 870 and A13 A14 are all fever [ sweat ] .",
            "What if the fever ? at least the game experience has stabilized at high frame rate , 90 or even 120 , TM 's 888 and 8gen1 is barely 60Hz , and it 's even hotter than others",
            "Your IQOO9 scheduling is quite conservative . My colleague 's Xiaomi Mi 12Pro is unhot every day . The manufacturers are conservatively scheduled . When running for a long time , the large nuclear frequency is extremely low [ DOGE ] .",
            "IQOO9 's heat dissipation stack and Xiaomi Mi 12Pro are not a level at all [ laughs without speaking ] .",
            "It 's almost okay , the area of the hot plate is almost the same ."
        ],
        "replies": [
            -1,
            0,
            1,
            2,
            0,
            4,
            5,
            0,
            7,
            8
        ],
        "speakers": [
            0,
            1,
            2,
            1,
            3,
            0,
            4,
            5,
            0,
            6
        ],
        "triplets": [
            [
                164,
                165,
                165,
                166,
                167,
                169,
                "neg",
                "IQOO9",
                "scheduling",
                "quite conservative"
            ],
            [
                205,
                206,
                207,
                210,
                215,
                218,
                "neg",
                "IQOO9",
                "heat dissipation stack",
                "not a level"
            ],
            [
                205,
                206,
                207,
                210,
                228,
                230,
                "other",
                "IQOO9",
                "heat dissipation stack",
                "almost okay"
            ],
            [
                205,
                206,
                232,
                237,
                238,
                239,
                "other",
                "IQOO9",
                "area of the hot plate",
                "almost"
            ],
            [
                96,
                97,
                133,
                135,
                136,
                140,
                "pos",
                "9",
                "game experience",
                "stabilized at high frame"
            ],
            [
                173,
                176,
                185,
                186,
                184,
                185,
                "neg",
                "Xiaomi Mi 12Pro",
                "scheduled",
                "conservatively"
            ],
            [
                212,
                214,
                207,
                210,
                215,
                218,
                "pos",
                "Mi 12Pro",
                "heat dissipation stack",
                "not a level"
            ],
            [
                173,
                176,
                195,
                198,
                199,
                201,
                "neg",
                "Xiaomi Mi 12Pro",
                "large nuclear frequency",
                "extremely low"
            ],
            [
                5,
                6,
                -1,
                -1,
                13,
                15,
                "pos",
                "IQOO9",
                "",
                "very fragrant"
            ],
            [
                173,
                176,
                -1,
                -1,
                177,
                178,
                "pos",
                "Xiaomi Mi 12Pro",
                "",
                "unhot"
            ],
            [
                96,
                97,
                -1,
                -1,
                101,
                102,
                "pos",
                "9",
                "",
                "fragrant"
            ],
            [
                112,
                113,
                -1,
                -1,
                120,
                121,
                "neg",
                "865",
                "",
                "fever"
            ],
            [
                114,
                115,
                -1,
                -1,
                120,
                121,
                "neg",
                "870",
                "",
                "fever"
            ],
            [
                116,
                117,
                -1,
                -1,
                120,
                121,
                "neg",
                "A13",
                "",
                "fever"
            ],
            [
                117,
                118,
                -1,
                -1,
                120,
                121,
                "neg",
                "A14",
                "",
                "fever"
            ]
        ],
        "targets": [
            [
                5,
                6,
                "IQOO9"
            ],
            [
                96,
                97,
                "9"
            ],
            [
                112,
                113,
                "865"
            ],
            [
                114,
                115,
                "870"
            ],
            [
                116,
                117,
                "A13"
            ],
            [
                117,
                118,
                "A14"
            ],
            [
                164,
                165,
                "IQOO9"
            ],
            [
                173,
                176,
                "Xiaomi Mi 12Pro"
            ],
            [
                205,
                206,
                "IQOO9"
            ],
            [
                212,
                214,
                "Mi 12Pro"
            ]
        ],
        "aspects": [
            [
                232,
                237,
                "area of the hot plate"
            ],
            [
                207,
                210,
                "heat dissipation stack"
            ],
            [
                165,
                166,
                "scheduling"
            ],
            [
                133,
                135,
                "game experience"
            ],
            [
                195,
                198,
                "large nuclear frequency"
            ],
            [
                185,
                186,
                "scheduled"
            ]
        ],
        "opinions": [
            [
                215,
                218,
                "not a level",
                "pos"
            ],
            [
                177,
                178,
                "unhot",
                "pos"
            ],
            [
                228,
                230,
                "almost okay",
                "neu"
            ],
            [
                167,
                169,
                "quite conservative",
                "neg"
            ],
            [
                120,
                121,
                "fever",
                "neg"
            ],
            [
                238,
                239,
                "almost",
                "neu"
            ],
            [
                13,
                15,
                "very fragrant",
                "pos"
            ],
            [
                215,
                218,
                "not a level",
                "neg"
            ],
            [
                199,
                201,
                "extremely low",
                "neg"
            ],
            [
                136,
                140,
                "stabilized at high frame",
                "pos"
            ],
            [
                184,
                185,
                "conservatively",
                "neg"
            ],
            [
                101,
                102,
                "fragrant",
                "pos"
            ]
        ]
        }
        self.assertEqual(self.base_diaasq.data[0], data)