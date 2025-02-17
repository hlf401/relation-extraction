import logging
import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))


# 定义tokenizer, 其他函数会使用MyTokenizer.bert_tokenizer and the funciton tokenize()
class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity

   
        logging.basicConfig(level=logging.WARNING) # Jupyter Notebook下执行需要重启kernel
        
        # 尝试产生一些日志
        logging.debug('Debug message')  # 不会被显示
        logging.info('Info message')    # 不会被显示
        logging.warning('Warning message')  # 不会被显示
        logging.error('Error message')  # 将会被显示
        logging.critical('Critical message')  # 将会被显示

    # item例子：{"h": {"name": "罗秀春", "pos": [17, 20]}, "t": {"name": "张默", "pos": [9, 11]}, "relation": "父母", "text": "-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长"}
    # 返回：tokenlize后的句子，和两个实体在新句子中的位置
    def tokenize(self, item):
        sentence = item['text']
        pos_head = item['h']['pos'] # [17, 20]
        pos_tail = item['t']['pos'] # [9, 11]
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail # [9, 11]
            pos_max = pos_head # [17, 20]
            rev = True  # 实体1和实体2在句子中的顺序 和 list中的 相反
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        # 整句：家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长
        # 分成三句sent0 sent1 sent2: 第一个实体前的为第一句；两个实体间的句子为第二句，第二个实体后的句子为第3句。
        # 两个实体提取：按在原句中的位置提取：ent0, ent1

       
        # sent0：-家庭张国立和父亲
        str_sent0 = sentence[:pos_min[0]]
        # "张默" 
        str_ent0 = sentence[pos_min[0]:pos_min[1]]
        # sent1：是张默和前妻
        str_sent1 = sentence[pos_min[1]:pos_max[0]]
        # "罗秀春"
        str_ent1 = sentence[pos_max[0]:pos_max[1]]
        # sent2：的儿子，据了解，张国立与罗女士相识于少年时代，长
        str_sent2 = sentence[pos_max[1]:]
        logging.debug("str_sent0:")
        logging.debug(str_sent0)
        logging.debug("str_sent1:")
        logging.debug(str_sent1)
        logging.debug("str_sent2:")
        logging.debug(str_sent2)       
        
        logging.debug("str_ent0:")
        logging.debug(str_ent0)
        logging.debug("str_ent1:")
        logging.debug(str_ent1)



        # sent0 sent1 sent2， ent0, ent1 都是用BertTokenizer.tokenize编码过的 
        sent0 = self.bert_tokenizer.tokenize(str_sent0)
        ent0 = self.bert_tokenizer.tokenize(str_ent0)
        sent1 = self.bert_tokenizer.tokenize(str_sent1)
        ent1 = self.bert_tokenizer.tokenize(str_ent1)
        sent2 = self.bert_tokenizer.tokenize(str_sent2)

        logging.debug("sent0:")
        logging.debug(sent0)
        logging.debug("sent1:")
        logging.debug(sent1)
        logging.debug("sent2:")
        logging.debug(sent2)       
        
        logging.debug("ent0:")
        logging.debug(ent0)
        logging.debug("ent1:")
        logging.debug(ent1)
        '''
        sent0:
        ['-', '家', '庭', '张', '国', '立', '和', '父', '亲']
        sent1:
        ['是', '张', '默', '和', '前', '妻']
        sent2:
        ['的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']
        ent0:
        ['张', '默']
        ent1:
        ['罗', '秀', '春']
        '''


        # 处理 mask 实体
        logging.debug("self.mask_entity:")
        logging.debug(self.mask_entity)

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2
        logging.debug("tokens after check self.mask_entity:")
        logging.debug(tokens)


        '''
        ['-', '家', '庭', '张', '国', '立', '和', '父', '亲', '张', '默', '是', '张', '默', '和', '前', '妻', '罗', '秀', '春', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']

        '''

        # 加入上述的分割字符串
        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        logging.debug("re_tokens after check self.mask_entity:")
        logging.debug(re_tokens)
        '''
        ['[CLS]', '-', '家', '庭', '张', '国', '立', '和', '父', '亲', '[unused2]', '张', '默', '[unused4]', '是', '张', '默', '和', '前', '妻', '[unused1]', '罗', '秀', '春', '[unused3]', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长', '[SEP]']

        '''

        # 返回值：
        # 去除开头和结尾字符
        logging.debug(re_tokens[1:-1])
        # ent0的新位置, 把[unused4]也算上了
        logging.debug(pos1)
        # ent2的新位置, 把[unused3]也算上了
        logging.debug(pos2)
        '''
        ['-', '家', '庭', '张', '国', '立', '和', '父', '亲', '[unused2]', '张', '默', '[unused4]', '是', '张', '默', '和', '前', '妻', '[unused1]', '罗', '秀', '春', '[unused3]', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']
        [20, 25]
        [10, 14]

        '''

        
        return re_tokens[1:-1], pos1, pos2


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask

# input_file: 如train_small.jsonl
# 返回值：
#       tokens_list：从input_file中读取的所有的tokens, 加入分隔符了
#       e1_mask_list：tokens_list对应的所有entry 1 的位置 标记为1，其他位置标记为0
#       e2_mask_list：tokens_list对应的所有entry 2 的位置 标记为1，其他位置标记为0
#       tags：tokens_list对应的所有关系tags
def read_data(input_file, tokenizer=None, max_len=128):
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line = line.strip()
            # line数据例子：{"h": {"name": "罗秀春", "pos": [17, 20]}, "t": {"name": "张默", "pos": [9, 11]}, "relation": "父母", "text": "-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长"}

            item = json.loads(line)
            if tokenizer is None:
                tokenizer = MyTokenizer()
            # tokens: 句子处理过后的 tokens
            # pos_e1: 实体1在新句子中的位置
            # pos_e2: 实体2在新句子中的位置
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                    pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                tokens_list.append(tokens)
                e1_mask = convert_pos_to_mask(pos_e1, max_len)
                e2_mask = convert_pos_to_mask(pos_e2, max_len)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                tag = item['relation']
                tags.append(tag)
    return tokens_list, e1_mask_list, e2_mask_list, tags


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


# 定义数据集
# __getitem__(self, idx): 获
#       返回idx对应的所有已经编码的数据
#       包含句子，2个实体，位置，关系等信息
class SentenceREDataset(Dataset):
    # data_file_path: 训练数据集，如datasets/train_small.jsonl
    # pretrained_model_path: 预训练模型，如pretrained_models/bert-base-chinese
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path

        # 加载tokenizer：主要使用BertTokenizer,   同时使用tokenize()函数将句子和实体分割 标识
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
        self.max_len = max_len

        # 读取训练集如datasets/train_small.jsonl所有数据：将原始entry1 entry2 relation, text数据转化为：
        # 1：self.tokenizer.tokenize()处理过的tokens_list
        # 2: self.e1_mask_list, self.e2_mask_list: 两个实体的在剧中的位置mask 数组 list
        # 3：self.tags： 两个实体之间的关系   list
        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)

        # 关系分类tags和index 列表
        self.tag2idx = get_tag2idx(self.tagset_path)
        logging.debug("SentenceREDataset() self.tag2idx:")
        logging.debug(self.tag2idx)
        '''
        SentenceREDataset() self.tag2idx:
        {'unknown': 0, '父母': 1, '夫妻': 2, '师生': 3, '兄弟姐妹': 4, '合作': 5, '情侣': 6, '祖孙': 7, '好友': 8, '亲戚': 9, '同门': 10, '上下级': 11}
        '''
    def __len__(self):
        return len(self.tags)

    # 返回idx对应的所有已经编码的数据
    # 包含句子，2个实体，位置，关系等信息
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_tag = self.tags[idx]

        # 编码idx对应句子的tokens
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']

        # 编码idx对应关系的tokens，直接用index
        sample_tag_id = self.tag2idx[sample_tag]

        # sample_e1_mask， sample_e2_mask已经对应数字数组；无需转化

        # 返回idx对应的所有已经编码的数据
        # 包含句子，2个实体，位置，关系等信息
        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }
        return sample
