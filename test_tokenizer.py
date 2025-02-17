import logging
import json
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from relation_extraction.data_utils import MyTokenizer, SentenceREDataset, get_idx2tag, load_checkpoint, save_checkpoint, convert_pos_to_mask

here = os.path.dirname(os.path.abspath(__file__))

def test_tokenizer():
    line = '{"h": {"name": "罗秀春", "pos": [17, 20]}, "t": {"name": "张默", "pos": [9, 11]}, "relation": "父母", "text": "-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长"}'
    item = json.loads(line)
    tokenizer = MyTokenizer()
    tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)

    logging.debug("pos_e1:")
    logging.debug(pos_e1)
    logging.debug("pos_e2:")
    logging.debug(pos_e2)

    # read_data()返回的数据
    
    max_len=128  # 句子最大字符数
    if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
            pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
        # tokens_list.append(tokens)
        e1_mask = convert_pos_to_mask(pos_e1, max_len)
        e2_mask = convert_pos_to_mask(pos_e2, max_len)
        # e1_mask_list.append(e1_mask)
        # e2_mask_list.append(e2_mask)
        tag = item['relation']
        # tags.append(tag)
    
        logging.debug("tokens:")
        logging.debug(tokens)
        logging.debug("e1_mask:")
        logging.debug(e1_mask)
        logging.debug("e2_mask:")
        logging.debug(e2_mask)
        logging.debug("tag:")
        logging.debug(tag)
        '''
        pos_e1:
        [20, 25]
        pos_e2:
        [10, 14]
        tokens:
        ['-', '家', '庭', '张', '国', '立', '和', '父', '亲', '[unused2]', '张', '默', '[unused4]', '是', '张', '默', '和', '前', '妻', '[unused1]', '罗', '秀', '春', '[unused3]', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']
        e1_mask:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        e2_mask:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tag:
        父母


        '''

def main():
    test_tokenizer()

if __name__ == '__main__':
    main()

