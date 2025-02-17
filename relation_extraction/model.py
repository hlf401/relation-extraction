import os
import logging
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

here = os.path.dirname(os.path.abspath(__file__))


# 定义下游任务模型
class SentenceRE(nn.Module):

    def __init__(self, hparams):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        # 载入BERT预训练模型
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3)


        # 隐藏层到最终分类
        # self.tagset_size =12 分类数目
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)



    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)

        logging.debug("forward() 0:")
        logging.debug("sequence_output.shape:")
        logging.debug(sequence_output.shape)
        logging.debug("pooled_output.shape:")
        logging.debug(pooled_output.shape)
        logging.debug("e1_mask.shape:")
        logging.debug(e1_mask.shape)
        logging.debug("e2_mask.shape:")
        logging.debug(e2_mask.shape)

        # 8: 每个批次8个数据
        # 128：每句最多128个字
        # 768：一个字对应768个特征值
        '''
        forward() 0:
        sequence_output.shape:
        torch.Size([8, 128, 768])
        pooled_output.shape:
        torch.Size([8, 768])
        
        e1_mask.shape:
        torch.Size([8, 128])
        e2_mask.shape:
        torch.Size([8, 128])

        '''

        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)


        logging.debug("forward() after entity_average():")
        logging.debug("e1_h.shape:")
        logging.debug(e1_h.shape)
        logging.debug("e2_h.shape:")
        logging.debug(e2_h.shape)

        '''
        forward() after entity_average():
        e1_h.shape:
        torch.Size([8, 768])
        e2_h.shape:
        torch.Size([8, 768])

        '''


        
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))
        logging.debug("forward() after activation():")
        logging.debug("e1_h.shape:")
        logging.debug(e1_h.shape)
        logging.debug("e2_h.shape:")
        logging.debug(e2_h.shape)
        '''
        forward() after activation():
        e1_h.shape:
        torch.Size([8, 768])
        e2_h.shape:
        torch.Size([8, 768])
        '''

        

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logging.debug("forward() after cat() and norm():")
        logging.debug("concat_h.shape:")
        logging.debug(concat_h.shape)
        '''
        forward() after cat() and norm():
        concat_h.shape:
        torch.Size([8, 2304])
        '''
        
        logits = self.hidden2tag(self.drop(concat_h))
        logging.debug("forward() after hidden2tag():")
        logging.debug("logits.shape:")
        logging.debug(logits.shape)
        '''
        forward() after hidden2tag():
        logits.shape:
        torch.Size([8, 12])
        
        '''

        return logits








    ''' 
    输入：
        hidden_output: [batch_size, j-i+1, dim] 
            [j-i+1, dim] ：实体字的个数, dim
            i,j代表位置
            
        e_mask: [batch_size, max_seq_len] 
            max_seq_len: 单个句子最长的长度
    返回值：
        [batch_size, dim]
    '''
    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]  
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
