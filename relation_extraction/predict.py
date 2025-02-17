import os
import re
import torch

from .data_utils import MyTokenizer, get_idx2tag, convert_pos_to_mask
from .model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))


def predict(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file

    # 保存的训练好的模型文件model.bin
    model_file = hparams.model_file

    # [id, tag] 列表
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = SentenceRE(hparams).to(device)

    
    # 载入训练好的模型
    model.load_state_dict(torch.load(model_file))

    # 进入预测模式
    model.eval()

    # load tokenlizer
    tokenizer = MyTokenizer(pretrained_model_path)

    
    while True:

        # 输入句子 和 要识别关系的2个实体
        text = input("输入中文句子：")
        entity1 = input("句子中的实体1：")
        entity2 = input("句子中的实体2：")


        # 2个实体必须在句子中出现
        match_obj1 = re.search(entity1, text)
        match_obj2 = re.search(entity2, text)
        if match_obj1 and match_obj2:  # 姑且使用第一个匹配的实体的位置
            e1_pos = match_obj1.span()
            e2_pos = match_obj2.span()

            # 把entry1 entry2 pos1 pos2写成如下形式，即数据集中的格式。
            item = {
                'h': {
                    'name': entity1,
                    'pos': e1_pos
                },
                't': {
                    'name': entity2,
                    'pos': e2_pos
                },
                'text': text
            }
            print("the input item is :")
            print(item)
            '''
            例子：
            句子：输入中文句子：哥哥和弟弟去北京
                句子中的实体1：哥哥
                句子中的实体2：弟弟
            转化为：
            {'h': {'name': '哥哥', 'pos': (0, 2)}, 't': {'name': '弟弟', 'pos': (3, 5)}, 'text': '哥哥和弟弟去北京'}
            '''


            # 返回：tokenlize后的句子，和两个实体在新句子中的位置
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)

            # 编码tokens
            encoded = tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')

            input_ids = encoded['input_ids'].to(device)
            token_type_ids = encoded['token_type_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # 根据pos_e1 标记位置数组
            e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(device)
            # 根据pos_e2 标记位置数组
            e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(device)


            
            with torch.no_grad():
                # 根据tokens entry and position 预测
                logits = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
                logits = logits.to(torch.device('cpu'))

            # logits.argmax(0).item() 预测的最有可能的结果    
            print("最大可能的关系是：{}".format(idx2tag[logits.argmax(0).item()]))

            # 所有可能的结果 按概率排序
            top_ids = logits.argsort(0, descending=True).tolist()
            for i, tag_id in enumerate(top_ids, start=1):
                print("No.{}：关系（{}）的可能性：{}".format(i, idx2tag[tag_id], logits[tag_id]))
        else:
            if match_obj1 is None:
                print('实体1不在句子中')
            if match_obj2 is None:
                print('实体2不在句子中')
