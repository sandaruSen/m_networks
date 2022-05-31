#  code with multiple positive negative pairs and triplet loss. considers all the negative expansions SEPERATELY. when getting the performance metrics the absolute value of distances is considered. not the cosie
#  input to the system: anchor, positive, negative
# change how the input is processed
import os
import json
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from transformers import AdamW, AutoModel, AutoTokenizer, BertTokenizer, BertModel
from datetime import datetime
from scipy.spatial import distance
import collections

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)


# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def seed_all(seed=42):
    """
    Fix seed for reproducibility
    """
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)


class config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_format = '%Y_%m_%d_%H_%M'
    SEED = 42
    KFOLD = 5
    # TRAIN_FILE = '../../data/AD_medal/csv_files/train.csv'
    # TRAIN_PAIR_FILE = '../../data/AD_medal/csv_files/train_pairs.csv'
    # EXTERNAL_PAIR_FILE = '../../data/AD_medal/csv_files/external_pairs.csv'
    # EXTERNAL_FILE = '../../data/AD_medal/csv_files/external_data.csv'
    # VAL_PAIR_FILE = '../data/AD_medal/csv_files/dev_pairs.csv'
    # VAL_FILE = '../../data/AD_medal/csv_files/dev.csv'

    TRAIN_FILE = '../data/AD/csv_files/train_pairs.csv'
    VAL_FILE = '../data/AD/csv_files/dev_pairs.csv'
    EXTERNAL_FILE = '../data/AD/csv_files/external_pairs.csv'

    ANCHOR_FILE = '../data/AD/anchor_sent.csv'
    ALL_ANCHOR_FILE = '../data/AD/all_anchor_sent.csv'
    SAVE_DIR = '../models/'
    MAX_LEN = 120
    freeze_layer_count = 10
    DROPOUT = 0.5
    INCLUDE_EXT = False
    MODEL = 'bert-base-uncased'
    # TOKENIZER = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    TOKENIZER = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    ALPHA = 0.5
    EPOCHS = 10
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    DICTIONARY = json.load(open('../data/AD/diction.json'))
    LEARNING_RATE = 2e-5
    OUTPUT_EMBED_SIZE = 64
    DETAILS_FILE = 'details.txt'
    A2ID = {}
    max_expansions = 0
    for k, v in DICTIONARY.items():
        if len(v) > max_expansions:
            max_expansions = len(v)
        for w in v:
            A2ID[w] = len(A2ID)
    print('Max no of expansions:', max_expansions)


class Dataset:
    def __init__(self, text, acronym, expansion, label, other_expansion, other_label, anchor_sentences, mode='train'):
        self.text = text
        self.acronym = acronym
        self.expansion = expansion
        self.label = label
        self.other_expansion = other_expansion
        self.other_label = other_label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.anchor_sentences = anchor_sentences
        self.mode = mode
        self.device = config.DEVICE

    def get_anchor_sentence(self, expansion):
        sents = self.anchor_sentences['anchor_sentence'].tolist()
        expansions = self.anchor_sentences['expansion'].tolist()
        index_ = expansions.index(expansion)
        value = sents[index_]

        # val = [expansion]
        # value = self.anchor_sentences.query('expansion in @val')["anchor_sentence"].iloc[0]
        return value

    def trim_ids(self, item):
        item['input_ids'] = item['input_ids'][:, :120]
        item['token_type_ids'] = item['token_type_ids'][:, :120]
        item['attention_mask'] = item['attention_mask'][:, :120]
        return item

    def __len__(self):
        return len(self.text)

    def sample_text(self, text, acronym, max_len):
        text = text.split()
        idx = text.index(acronym)
        left_idx = max(0, idx - max_len // 2)
        right_idx = min(len(text), idx + max_len // 2)
        sampled_text = text[left_idx:right_idx]
        return ' '.join(sampled_text)

    def trim_sentence(self, text, expansion, max_len):
        text = text.split()
        expansions = expansion.split()
        exp_len = len(expansions)
        start_idx = text.index(expansions[0])
        end_idx = text.index(expansions[-1])
        max_len = max_len - exp_len
        left_idx = max(0, start_idx - max_len // 2)
        right_idx = min(len(text), end_idx + max_len // 2)
        sampled_text = text[left_idx:start_idx] + expansions + text[end_idx:right_idx]
        return ' '.join(sampled_text)

    def process_data_(self, text, acronym, expansion, tokenizer, max_len, other_expansion, anchor_sentence, mode):
        text = str(text)
        expansion = str(expansion)
        acronym = str(acronym)
        other = str(other_expansion)
        anchor_sentence = str(anchor_sentence)

        n_tokens = len(anchor_sentence.split())
        if n_tokens > 120:
            anchor_sentence = self.sample_text(anchor_sentence, acronym, config.MAX_LEN)

        # anchor sentence -- this is from the create list of anchor sentences - contains the acronym
        anchor_token_ids = tokenizer(anchor_sentence, padding="max_length", max_length=config.MAX_LEN,
                                     return_tensors="pt")

        if mode == 'train':

            #  positive sentence -- acornym in the sample sentence is replaced by the correct expansion
            positive_sentence = text.replace(acronym, expansion)
            n_positive_tokens = len(positive_sentence.split())
            if n_positive_tokens > config.MAX_LEN:
                positive_sentence = self.trim_sentence(positive_sentence, expansion, config.MAX_LEN)
            positive_token_ids = tokenizer(positive_sentence, padding="max_length", max_length=config.MAX_LEN,
                                           return_tensors="pt")

            #  negative  sentence -- acornym in the sample sentence is replaced by the incorrect expansion
            negative_sentence = text.replace(acronym, other_expansion)
            n_negative_tokens = len(negative_sentence.split())
            if n_negative_tokens > config.MAX_LEN:
                negative_sentence = self.trim_sentence(negative_sentence, other_expansion, config.MAX_LEN)
            other_token_ids = tokenizer(negative_sentence, padding="max_length", max_length=config.MAX_LEN,
                                        return_tensors="pt")
        else:
            # in the validation stage sentences from anchor list is not required as we consider the sample sentence with acronym as the anchor sentence
            #  anchor sentence -- acornym in the sample sentence without any replacements
            positive_sentence = text
            n_positive_tokens = len(positive_sentence.split())
            if n_positive_tokens > 120:
                positive_sentence = self.sample_text(positive_sentence, acronym, config.MAX_LEN)
            positive_token_ids = tokenizer(positive_sentence, padding="max_length", max_length=config.MAX_LEN,
                                           return_tensors="pt")

            other = other_expansion
            other.append(expansion)
            other_list = set(other)
            other = list(other_list)
            other_token_ids = []
            for i in range(len(other)):
                #  other sentences -- acornym in the sample sentence is replaced by  expansions
                other_sentence = text.replace(acronym, other[i])
                n_other_tokens = len(other_sentence.split())
                if n_other_tokens > config.MAX_LEN:
                    other_sentence = self.trim_sentence(other_sentence, other[i], config.MAX_LEN)
                o_token_ids = tokenizer(other_sentence, padding="max_length", max_length=config.MAX_LEN,
                                        return_tensors="pt")
                other_token_ids.append(o_token_ids)
        return {
            'ids': positive_token_ids,
            'text': text,
            'expansion': expansion,
            'acronym': acronym,
            'other_ids': other_token_ids,
            'other_expansion': other,
            'anchor_sentence': anchor_sentence,
            'anchor_token_ids': anchor_token_ids,
            'others': other
        }

    def __getitem__(self, item):
        anchor_sentence = self.get_anchor_sentence(self.expansion[item])
        # exp = self.other_expansion[item]
        data = self.process_data_(
            self.text[item],
            self.acronym[item],
            self.expansion[item],
            self.tokenizer,
            self.max_len,
            self.other_expansion[item],
            anchor_sentence,
            self.mode
        )
        if data['ids']['input_ids'].shape[1] > 120:
            data['ids'] = self.trim_ids(data['ids'])
        if data['anchor_token_ids']['input_ids'].shape[1] > 120:
            data['anchor_token_ids'] = self.trim_ids(data['anchor_token_ids'])
        if self.mode == 'train' and data['other_ids']['input_ids'].shape[1] > 120:
            data['other_ids'] = self.trim_ids(data['other_ids'])

        if self.mode == 'test':
            no_possible_expansions = len(data['other_ids'])
            no_of_expansions = config.max_expansions

            other_ids = []
            other_token_type_ids = []
            other_attention_mask = []
            for j in range(len(data['other_ids'])):
                if data['other_ids'][j]['input_ids'].shape[1] > 120:
                    data['other_ids'][j] = self.trim_ids(data['other_ids'][j])
                other_ids.append(torch.squeeze(data['other_ids'][j]['input_ids']))
                other_token_type_ids.append(torch.squeeze(data['other_ids'][j]['token_type_ids']))
                other_attention_mask.append(torch.squeeze(data['other_ids'][j]['attention_mask']))

            if no_of_expansions > no_possible_expansions:
                padding = [torch.tensor(([0] * self.max_len)) for i in range(no_of_expansions - no_possible_expansions)]
                data['other_ids'] = other_ids + padding
                data['other_mask'] = other_attention_mask + padding
                data['other_token_type'] = other_token_type_ids + padding
            else:
                data['other_ids'] = other_ids
                data['other_mask'] = other_attention_mask
                data['other_token_type'] = other_token_type_ids
            # possible_expansions = '###'.join(self.other_expansion[item])
            possible_expansions = '###'.join(data['others'])
            # possible_expansions = possible_expansions + "###" + self.expansion[item]

            #######################################check this#############################
            test_data = []
            if len(data['other_ids']) > config.max_expansions or len(
                    data['other_token_type']) > config.max_expansions or len(
                data['other_mask']) > config.max_expansions:
                print(len(data['other_ids']))
                print(self.text[item])
            test_data.append(torch.stack(data['other_ids']))
            test_data.append(torch.stack(data['other_token_type']))
            test_data.append(torch.stack(data['other_mask']))
            # test_data = torch.tensor(test_data, dtype=torch.long)
            no_possible_expansions = torch.tensor(no_possible_expansions)
            return data['anchor_token_ids'], data['ids'], test_data, no_possible_expansions, self.expansion[
                item], possible_expansions

        return {
            'ids': data['ids'],
            'text': data['text'],
            'expansion': data['expansion'],
            'acronym': data['acronym'],
            'other_ids': data['other_ids'],
            'other_expansion': data['other_expansion'],
            'label': torch.tensor(self.label[item], dtype=torch.long).to(self.device),
            'other_label': torch.tensor(self.other_label[item], dtype=torch.long).to(self.device),
            'anchor_sentence': data['anchor_sentence'],
            'anchor_token_ids': data['anchor_token_ids'],
        }


class BertAD(nn.Module):
    def __init__(self):
        super(BertAD, self).__init__()
        # self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        #  self.layer_1 = nn.Linear(768 * config.MAX_LEN, 100)
        # self.bert = AutoModel.from_pretrained("xhlu/electra-medal")
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.layer_1 = nn.Linear(768 * config.MAX_LEN, config.OUTPUT_EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward_one(self, ids, mask, token_type):
        output = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type)
        # output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2]))
        # .last_hidden_state
        # [:,0,:] - [all sentences, only the first position cls : all hidden states]   ----- [0][:, 0,:]

        output = output.last_hidden_state
        output = torch.reshape(output, (output.shape[0], output.shape[1] * output.shape[2]))
        output = self.dropout(output)
        output = self.layer_1(output)
        return output

    def forward(self, anchor_token_ids, anchor_mask, anchor_token_type,
                token_ids, token_mask, token_type,
                other_ids=None, other_mask=None, other_token_type=None, mode='train', max_lengths=0):
        if mode == 'train':
            output_anchor = self.forward_one(anchor_token_ids, anchor_mask, anchor_token_type)

            output_positive = self.forward_one(token_ids, token_mask, token_type)

            output_negative = self.forward_one(other_ids, other_mask, other_token_type)

            return output_anchor, output_positive, output_negative
        else:
            anchor_embeddings = []
            for i in range(max_lengths):
                _ids = other_ids[:, i, :]
                _mask = other_mask[:, i, :]
                _type = other_token_type[:, i, :]
                output = self.forward_one(_ids, _mask, _type)
                output = torch.unsqueeze(output, dim=1)
                if i == 0:
                    anchor_embeddings = output
                else:
                    anchor_embeddings = torch.cat((anchor_embeddings, output), dim=1)
            # in the validation and test stages, the anchor sentece is the positive sample with  no replacement
            output_ = self.forward_one(token_ids, token_mask, token_type)
            return anchor_embeddings, output_


def triplet_loss(anchor, positive, negative, alpha=config.ALPHA, device='cuda:0'):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    embedding -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    #     print('embedding.shape = ',embedding)

    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    #     anchor = embedding[:,0,:]
    #     positive = embedding[:,1,:]
    #     negative = embedding[:,2,:]

    # distance between the anchor and the positive
    pos_dist = torch.sum((anchor - positive).pow(2), axis=1)

    # distance between the anchor and the negative
    neg_dist = torch.sum((anchor - negative).pow(2), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = torch.max(basic_loss, torch.tensor([0], device=device).float())

    return loss, pos_dist, neg_dist


def compute_metrics(prediction):
    true = [1] * len(prediction)
    precision = precision_score(true, prediction)
    recall = recall_score(true, prediction)
    fscore = f1_score(true, prediction)
    return precision, recall, fscore


def compute_distances_for_valid(output_anchors, output_positive, lengths, max_lengths):
    lengths = lengths.detach().cpu().numpy().tolist()

    for i in range(max_lengths):
        output = torch.sum((output_anchors[:, i, :] - output_positive).pow(2), axis=1)
        output = torch.unsqueeze(output, dim=1)
        if i == 0:
            cos_ = output
        else:
            cos_ = torch.cat([cos_, output], dim=1)

    cos_values = cos_.detach().cpu().numpy().tolist()
    values = [cos_values[i][:lengths[i]] for i in range(len(cos_values))]
    min_values = [min(item) for item in values]
    min_indices = [values[i].index(min_values[i]) for i in range(len(values))]
    return min_indices


class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()
    predictions = []

    for d in data_loader:
        ids = d['ids']  # torch.Size([12, 192])
        other_ids = d['other_ids']
        other_expansion = d['other_expansion']
        label = d['label']
        other_label = d['other_label']
        anchor_token_ids = d['anchor_token_ids']

        # other_ids =  to_device(other_ids)
        anchor_ids = torch.squeeze(anchor_token_ids['input_ids'].to(device))
        anchor_token_type_ids = torch.squeeze(anchor_token_ids['token_type_ids'].to(device))
        anchor_attention_mask = torch.squeeze(anchor_token_ids['attention_mask'].to(device))

        positive_ids = torch.squeeze(ids['input_ids'].to(device))
        positive_token_type_ids = torch.squeeze(ids['token_type_ids'].to(device))
        positive_attention_mask = torch.squeeze(ids['attention_mask'].to(device))

        negative_ids = torch.squeeze(other_ids['input_ids'].to(device))
        negative_token_type_ids = torch.squeeze(other_ids['token_type_ids'].to(device))
        negative_attention_mask = torch.squeeze(other_ids['attention_mask'].to(device))

        model.zero_grad()
        output_anchor, output_positive, output_negative = model(anchor_ids, anchor_attention_mask,
                                                                anchor_token_type_ids,
                                                                positive_ids, positive_attention_mask,
                                                                positive_token_type_ids,
                                                                negative_ids, negative_attention_mask,
                                                                negative_token_type_ids)
        loss, pos_dist, neg_dist = triplet_loss(output_anchor, output_positive, output_negative, device=device)

        loss.mean().backward()
        optimizer.step()

        losses.update(loss.mean().item(), anchor_ids.size(0))

        pos_dist = pos_dist.detach().cpu().numpy().tolist()
        neg_dist = neg_dist.detach().cpu().numpy().tolist()
        prediction = [1 if pos_dist[i] < neg_dist[i] else 0 for i in range(len(pos_dist))]

        # prediction = compute_cosine_similarity(output_anchor, output_positive, output_negative)
        predictions.extend(prediction)
    precision, recall, fscore = compute_metrics(predictions)

    return fscore, precision, recall, losses.avg, model


def eval_fn(data_loader, model, device):
    model.eval()

    predictions = []
    for anchor_token_ids, ids, d, lengths, correct_expansion, possible_expansions in data_loader:
        ids = ids  # torch.Size([3, 192])
        anchor_token_ids = anchor_token_ids
        others = d

        other_ids = others[0]
        other_token_type = others[1]
        other_mask = others[2]

        max_lenghts = torch.max(lengths)
        max_lenghts = max_lenghts.to(device)

        anchor_ids = torch.squeeze(anchor_token_ids['input_ids'].to(device))
        anchor_token_type_ids = torch.squeeze(anchor_token_ids['token_type_ids'].to(device))
        anchor_attention_mask = torch.squeeze(anchor_token_ids['attention_mask'].to(device))

        positive_ids = torch.squeeze(ids['input_ids'].to(device))
        positive_token_type_ids = torch.squeeze(ids['token_type_ids'].to(device))
        positive_attention_mask = torch.squeeze(ids['attention_mask'].to(device))

        negative_ids = torch.squeeze(other_ids.to(device))
        negative_token_type_ids = torch.squeeze(other_token_type.to(device))
        negative_attention_mask = torch.squeeze(other_mask.to(device))

        with torch.no_grad():

            anchor_embeddings, output_positive = model(anchor_ids, anchor_attention_mask, anchor_token_type_ids,
                                                       positive_ids, positive_attention_mask, positive_token_type_ids,
                                                       negative_ids, negative_attention_mask, negative_token_type_ids,
                                                       mode='test', max_lengths=max_lenghts)
            min_indices = compute_distances_for_valid(anchor_embeddings, output_positive, lengths, max_lenghts)

            for i in range(len(other_ids)):
                possible_expansions_ = possible_expansions[i].split("###")
                predicted_expansion = possible_expansions_[min_indices[i]]
                if predicted_expansion == correct_expansion[i]:
                    prediction = 1
                    predictions.append(prediction)
                else:
                    prediction = 0
                    predictions.append(prediction)

    precision, recall, fscore = compute_metrics(predictions)

    return fscore, precision, recall


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(df_train, df_val, df_test, fold, df_anchor_sent):
    train_dataset = Dataset(
        text=df_train.text.values,
        acronym=df_train.acronym_.values,
        expansion=df_train.expansion.values,
        label=df_train.label.values,
        other_expansion=df_train.neg_expansion.values,
        other_label=df_train.other_label.values,
        anchor_sentences=df_anchor_sent
    )

    valid_dataset = Dataset(
        text=df_val.text.values,
        acronym=df_val.acronym_.values,
        expansion=df_val.expansion.values,
        label=df_val.label.values,
        other_expansion=df_val.other_expansion.values,
        other_label=df_val.other_label.values,
        anchor_sentences=df_anchor_sent,
        mode='test'
    )
    test_dataset = Dataset(
        text=df_test.text.values,
        acronym=df_test.acronym_.values,
        expansion=df_test.expansion.values,
        label=df_test.label.values,
        other_expansion=df_test.other_expansion.values,
        other_label=df_test.other_label.values,
        anchor_sentences=df_anchor_sent,
        mode='test'
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True
    )
    model = BertAD()
    # print(model)
    if config.freeze_layer_count:
        # We freeze here the embeddings of the model
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        if config.freeze_layer_count != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in model.bert.encoder.layer[:config.freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    with open(details_file_path, 'a') as file:
        file.write(
            f'\nNo of Model Parameters: {str(count_parameters(model))} \n')

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)

    # es = EarlyStopping(patience=2, mode="max")

    print('Starting training....')
    best_valid_f1 = 0
    for epoch in range(config.EPOCHS):
        train_f1, train_precision, train_recall, train_loss, model_ = train_fn(train_data_loader, model, optimizer,
                                                                               device)
        valid_f1, valid_precision, valid_recall = eval_fn(valid_data_loader, model_, device)
        valid_loss = 0
        print(
            f'Fold {fold} | Epoch :{epoch + 1}  | Train loss :{train_loss} | Train precision :{train_precision} | Train recall  :{train_recall} | Train f1 :{train_f1}')

        print(
            f'Fold {fold} | Epoch :{epoch + 1}  Validation loss :{valid_loss} | Validation precision :{valid_precision} | Validation recall  :{valid_recall} | Validation f1 :{valid_f1}')
        with open(details_file_path, 'a') as file:
            file.write(
                f'\nFold {fold} | Epoch :{epoch + 1}  | Train loss :{train_loss} | Train precision :{train_precision} | Train recall  :{train_recall} | Train f1 :{train_f1}')

            file.write(
                f'\nFold {fold} | Epoch :{epoch + 1}   | Validation loss :{valid_loss} | Validation precision :{valid_precision} | Validation recall  :{valid_recall} | Validation f1 :{valid_f1}')

        if fold is None:
            path = config.SAVE_DIR + timestamp + 'model.pth'
            torch.save(model_.state_dict(), path)
            # es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, "model.bin"))
        elif valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            path = config.SAVE_DIR + timestamp + 'best_model.pth'
            torch.save(model_.state_dict(), path)
        else:
            path = config.SAVE_DIR + timestamp + 'model_' + str(fold) + '.pth'
            torch.save(model_.state_dict(), path)
            # es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR + timestamp, f"model_{fold}.bin"))
        # if es.early_stop:
        #     break

    test_f1, test_precision, test_recall = eval_fn(test_data_loader, model_, device)
    print(
        f'Test precision :{test_precision} | Test recall  :{test_recall} | Test f1 :{test_f1}')
    with open(details_file_path, 'a') as file:
        file.write(
            f'\nTest precision :{test_precision} | Test recall  :{test_recall} | Test f1 :{test_f1}')


    # return es.best_score
    return valid_f1


def get_expansions(row):
    others = config.DICTIONARY[row['acronym_']].copy()
    others.remove(row['expansion'])
    return others


def set_label(df):
    df['label'] = 1
    df['other_expansion'] = df.apply(get_expansions, axis=1)
    df['other_label'] = 0
    return df


def run_k_fold(fold_id):
    seed_all()

    df_anchor_sent = pd.read_csv(config.ALL_ANCHOR_FILE)

    df_train = pd.read_csv(config.TRAIN_FILE)
    df_val = pd.read_csv(config.VAL_FILE)

    df_train = set_label(df_train)
    df_val = set_label(df_val)
    df_test = df_val

    if config.INCLUDE_EXT == True:
        df_ext = pd.read_csv(config.EXTERNAL_FILE)
        df_ext = set_label(df_ext)
        train = pd.concat([df_train, df_ext], ignore_index=True)
        df_train = train


    return run(df_train, df_val, df_test, fold_id, df_anchor_sent)


if __name__ == '__main__':
    device = config.DEVICE
    timestamp = datetime.now().strftime(config.time_format) + '/'
    os.makedirs(config.SAVE_DIR + timestamp, exist_ok=True)
    model_file_path = config.SAVE_DIR + timestamp
    details_file_path = config.SAVE_DIR + timestamp + config.DETAILS_FILE
    with open(details_file_path, 'a') as file:
        file.write('File Name:  Disambiguation 6  ******** with processing changed for AAAI AD data')
        file.write(f'\nStart time:{timestamp} ')
        file.write(
            '\nConsiders anchor positive negative (only one negative) at a time. Changed how the input was processed.')
        file.write(
            '\nIn the training stage, anchor sentence: respective anchor sentence with the acronym. expansion is not in the sentence. | positive sentence: sample sentence with the acronym replaced with correct expansion. | Negative sentence: sample sentence with the acronym replaced with an incorrect expansion')
        file.write(
            '\nIn the validation stage, anchor sentence: sample sentence with the acronym. | Negative or other sentences: list of  sample sentences with acronym replaced with all possible expansions\n')
        file.write(
            f'\nNo. of epochs:  {config.EPOCHS} |  Batch size :{config.TRAIN_BATCH_SIZE} | Kfold :{config.KFOLD} | Max len  :{config.MAX_LEN} | Learning Rate :{config.LEARNING_RATE} | Dropout :{config.DROPOUT} | Freeze layer count :{config.freeze_layer_count} | Outpu embed size :{config.OUTPUT_EMBED_SIZE} | ALPHA for triplet loss :{config.ALPHA} | Includes external data for training :{config.INCLUDE_EXT}\n')
    try:
        f0 = run_k_fold(0)
        f = [f0]
        print(f'\nFinal of F score:. {np.mean(f)}')
        end_timestamp = datetime.now().strftime(config.time_format)
        with open(details_file_path, 'a') as file:
            file.write(
                f'\nNo. of folds:  {len(f)} | Final of F score:  {np.mean(f)}')
            file.write(f'\nEnd time:{end_timestamp}')
    except Exception as e:
        print(e)
        with open(details_file_path, 'a') as file:
            file.write('\n\n')
            file.write(str(e))

    # create_positive_negative_pairs()

#         anchor_sentences = create_anchor_sentences()
