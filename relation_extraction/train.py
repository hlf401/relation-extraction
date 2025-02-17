import logging
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from .data_utils import SentenceREDataset, get_idx2tag, load_checkpoint, save_checkpoint
from .model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))

# 使用测试数据集进行模型训练
# 使用验证数据集进行评价
def train(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    max_len = hparams.max_len
    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay


    logging.debug("The training hparams are:")
    logging.debug(hparams)
    '''
       Namespace(
       checkpoint_file='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../saved_models\\checkpoint.json', 
       device='cuda', 
       dropout=0.1, 
       embedding_dim=768, 
       epochs=20, 
       learning_rate=1e-05, 
       log_dir='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../saved_models\\runs', 
       max_len=128, 
       model_file='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../saved_models\\model.bin', 
       output_dir='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../saved_models', 
       pretrained_model_path='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../pretrained_models/bert-base-chinese', 
       seed=12345, 
       tagset_file='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../datasets/relation.txt', 
       train_batch_size=8, 
       train_file='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../datasets/train_small.jsonl', 
       validation_batch_size=8, 
       validation_file='D:\\code_new\\relation-extraction_hlf\\relation-extraction\\relation_extraction\\../datasets/val_small.jsonl', 
       weight_decay=0)   
    '''

    # 加载数据集：train_dataset， 会对数据集进行tokenize, 预处理，和编码
    train_dataset = SentenceREDataset(train_file, tagset_path=tagset_file,
                                      pretrained_model_path=pretrained_model_path,
                                      max_len=max_len)

    # DataLoader批量迭代数据集        
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    # 从tagset_file中读出所有关系类型(idx, tag)列表
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    logging.debug("tagset_size is: %d" % hparams.tagset_size)
    logging.debug(idx2tag)
    '''
    tagset_size is: 12
    {0: 'unknown', 1: '父母', 2: '夫妻', 3: '师生', 4: '兄弟姐妹', 5: '合作', 6: '情侣', 7: '祖孙', 8: '好友', 9: '亲戚', 10: '同门', 11: '上下级'}
    '''
    
    model = SentenceRE(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    # log writter
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


    # 训练
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            #  读出一批次数据
            token_ids = sample_batched['token_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_mask'].to(device)
            e2_mask = sample_batched['e2_mask'].to(device)
            tag_ids = sample_batched['tag_id'].to(device)

            
            model.zero_grad()
            # 使用SentenceRE.forward()进行预测（刚开始随机参数计算）
            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)

            # logits对比实际正确值tag_ids，进行参数调整
            loss = criterion(logits, tag_ids)
            loss.backward()
            optimizer.step()

            # 计算错误率
            running_loss += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
                running_loss = 0.0


        # 如果有验证数据集的话，每个epoch，验证一批数据，并输出评价值
        if validation_file:
            # 加载验证数据集：validation_dataset， 会对数据集进行tokenize, 预处理，和编码
            validation_dataset = SentenceREDataset(validation_file, tagset_path=tagset_file,
                                                   pretrained_model_path=pretrained_model_path,
                                                   max_len=max_len)
            val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                tags_true = []
                tags_pred = []
                for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):

                    #  读出一批次数据
                    token_ids = val_sample_batched['token_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)
                    e1_mask = val_sample_batched['e1_mask'].to(device)
                    e2_mask = val_sample_batched['e2_mask'].to(device)
                    tag_ids = val_sample_batched['tag_id']

                    
                    logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)

                    # 输出预测的tag id
                    pred_tag_ids = logits.argmax(1)

                    # 实际tag_id列表
                    tags_true.extend(tag_ids.tolist())
                    # 预测的tag_id列表
                    tags_pred.extend(pred_tag_ids.tolist())


                # 输出和保存评价指标
                print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))
                f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
                precision = metrics.precision_score(tags_true, tags_pred, average='weighted')
                recall = metrics.recall_score(tags_true, tags_pred, average='weighted')
                accuracy = metrics.accuracy_score(tags_true, tags_pred)
                writer.add_scalar('Validation/f1', f1, epoch)
                writer.add_scalar('Validation/precision', precision, epoch)
                writer.add_scalar('Validation/recall', recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}

                # 如果f1 比较好，则保存、更新当前训练好的模型
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)
                save_checkpoint(checkpoint_dict, checkpoint_file)

    writer.close()
