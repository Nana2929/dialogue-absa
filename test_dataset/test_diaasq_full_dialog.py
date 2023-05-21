'''
@File    :   test_diaasq_full_diag.py
@Time    :   2023/05/19 21:37:40
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   test full dialogue dataset
@Ref     :   https://openhome.cc/Gossip/CodeData/PythonTutorial/UnitTestPy3.html
'''
import unittest
from typing import Dict
from dataset.diaasq.full_dialogue_dataset import FullDiaAsqDataset

class TestFullDiaAsq(unittest.TestCase):
    # set self.maxDiff = None to see full diff
    maxDiff = None
    def setUp(self):
        common_args = {
            'train_split_name': 'train',
            'test_split_name': 'valid',
            'data_root': 'data/diaasq/dataset',
            'k': 1, # any
            'seed': 0,
            'prompt_path': f'./prompt/experiment/diaasq-fulldialog'
        }
        self.dataset_en = FullDiaAsqDataset(src='en', **common_args)
        self.dataset_zh = FullDiaAsqDataset(src='zh', **common_args)

    def tearDown(self):
        self.dataset_en = None
        self.dataset_zh = None

    @staticmethod
    def strip_(text):
        return text.replace(" ", "").replace("\n", "").replace("\t", "")

    def test_zh_form_example(self):
        expected_output = """
        輸入：
        A: 这手机不怎么滴但是对比iPhone我觉得除了处理器其他都比iPhone好[笑cry]
        B: iPhone就处理器和ios出色,其他真的多年被安卓吊打
        C: 确实。销量也吊打安卓。安卓厂商天天自称高端高端,然而在苹果面前就是个孩子[允悲]
        B: 三星,小米不都超过苹果?因为安卓系统选择太多,ios只有一家,如果安卓也只有一家,你觉得结果是什么?
        D: 随你说咯,反正我没用过小米,也不能妄加评论。只是外出旅游,朋友小米手机从来没见过拍得好的照片过。特别这次去马岭河瀑布,必须要抓拍,兄弟的米11张张糊的,这体验也没谁了
        A: 小米11拍照确实不行啊[黑线][黑线][黑线]
        D: 参数年年吊打,体验年年勉强..就这么回事儿。手机是自己的,谁用谁知道。
        A: 就算是体验也比iPhone好啊我可受不了iPhone落后的影像系统

        輸出：

        """
        formatted_test_sample = self.dataset_zh._form_example(self.dataset_zh.data[0], with_ans = False)
        self.assertEqual(self.strip_(formatted_test_sample), self.strip_(expected_output))

    def test_en_form_example(self):
        expected_output = """
        Input:
        A: This phone is not very good , but compared to the iPhone , I think it is better than the iPhone except for the processor [ laughs cry ]
        B: The iPhone is excellent as the processor and iOS , and others have been beaten by Android for many years .
        C: really . Sales also beat Android . Android manufacturers claim to be high - end and high - end every day , but they are just children in front of Apple .
        B: Samsung , Xiaomi does not all exceed Apple?Because there are too many Android systems , there is only one iOS . If there is only one Android , what do you think of the result ?
        D: As you say , I have n't used Xiaomi , so I ca n't comment . But traveling , my friend 's Xiaomi phone never took good photos . Especially when went to Malinghe Waterfall this time , we had to take pictures . Every photo taken by my brother 's Mi 11 was blurry . This experience is also speechless .
        A: Xiaomi 11 is really not good [ black line ] [ black line ] [ black line ] .
        D: The parameters overwhelme every year , and the experience is general every year ... that 's all . The phone is yours , who uses it , who knows .
        A: Even the experience is better than the iPhone . I ca n't stand the backward image system of the iPhone .

        Output:
        """
        formatted_test_sample = self.dataset_en._form_example(self.dataset_en.data[0], with_ans = False)
        self.assertEqual(self.strip_(formatted_test_sample), self.strip_(expected_output))
