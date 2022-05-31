import os
import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import model_selection
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from transformers import AdamW, AutoModel, AutoTokenizer, BertTokenizer, BertModel
import tokenizers
from datetime import datetime


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
    # -------------------------------------------------------
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_format = '%Y_%m_%d_%H_%M'
    SEED = 42
    KFOLD = 5

    TRAIN_FILE = '../data/AD/csv_files/train.csv'
    VAL_FILE = '../data/AD/csv_files/dev.csv'
    EXTERNAL_FILE = '../data/AD/csv_files/external_data.csv'
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
    OUTPUT_EMBED_SIZE = 64
    DICTIONARY = json.load(open('../data/AD/diction.json'))
    LEARNING_RATE = 1e-5
    DETAILS_FILE = 'details.txt'
    A2ID = {}
    max_expansions = 0
    for k, v in DICTIONARY.items():
        if len(v) > max_expansions:
            max_expansions = len(v)
        for w in v:
            A2ID[w] = len(A2ID)
    # print('Max no of expansions:', max_expansions)


def sample_text(text, acronym, max_len):
    text = text.split()
    idx = text.index(acronym)
    left_idx = max(0, idx - max_len // 2)
    right_idx = min(len(text), idx + max_len // 2)
    sampled_text = text[left_idx:right_idx]
    return ' '.join(sampled_text)


def process_data(text, acronym, expansion, tokenizer, max_len):
    text = str(text)
    expansion = str(expansion)
    acronym = str(acronym)

    n_tokens = len(text.split())
    if n_tokens > 120:
        text = sample_text(text, acronym, 120)

    answers = acronym + ' ' + ' '.join(
        config.DICTIONARY[acronym])  # eg answer:  'MGM markov geographic model manifold geometry matching'
    start = answers.find(expansion)  # 4
    end = start + len(expansion)  # 26

    char_mask = [0] * len(answers)
    for i in range(start, end):
        char_mask[i] = 1

    tok_answer = tokenizer.encode(answers)
    try:
        answer_ids = tok_answer.ids
        answer_offsets = tok_answer.offsets
    except:
        tok_answer = tokenizer.encode_plus(answers)
        answer_ids = tok_answer.encodings[0].ids
        answer_offsets = tok_answer.encodings[0].offsets

    answer_ids = answer_ids[1:-1]
    answer_offsets = answer_offsets[1:-1]

    target_idx = []
    for i, (off1, off2) in enumerate(answer_offsets):
        if sum(char_mask[off1:off2]) > 0:
            target_idx.append(i)

    start = target_idx[0]
    end = target_idx[-1]

    try:
        text_ids = tokenizer.encode(text).ids[1:-1]
    except:
        text_ids = tokenizer.encode_plus(text).encodings[0].ids[1:-1]

    token_ids = [101] + answer_ids + [102] + text_ids + [102]
    offsets = [(0, 0)] + answer_offsets + [(0, 0)] * (len(text_ids) + 2)
    mask = [1] * len(token_ids)
    token_type = [0] * (len(answer_ids) + 1) + [1] * (2 + len(text_ids))

    text = answers + ' ' + text
    start = start + 1
    end = end + 1

    padding = max_len - len(token_ids)

    if padding >= 0:
        token_ids = token_ids + ([0] * padding)
        token_type = token_type + [1] * padding
        mask = mask + ([0] * padding)
        offsets = offsets + ([(0, 0)] * padding)
    else:
        token_ids = token_ids[0:max_len]
        token_type = token_type[0:max_len]
        mask = mask[0:max_len]
        offsets = offsets[0:max_len]

    assert len(token_ids) == max_len
    assert len(mask) == max_len
    assert len(offsets) == max_len
    assert len(token_type) == max_len

    return {
        'ids': token_ids,
        'mask': mask,
        'token_type': token_type,
        'offset': offsets,
        'start': start,
        'end': end,
        'text': text,
        'expansion': expansion,
        'acronym': acronym,
    }


class Dataset:
    def __init__(self, text, acronym, expansion):
        self.text = text
        self.acronym = acronym
        self.expansion = expansion
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.acronym[item],
            self.expansion[item],
            self.tokenizer,
            self.max_len,

        )

        return {
            'ids': torch.tensor(data['ids'], dtype=torch.long),
            'mask': torch.tensor(data['mask'], dtype=torch.long),
            'token_type': torch.tensor(data['token_type'], dtype=torch.long),
            'offset': torch.tensor(data['offset'], dtype=torch.long),
            'start': torch.tensor(data['start'], dtype=torch.long),
            'end': torch.tensor(data['end'], dtype=torch.long),
            'text': data['text'],
            'expansion': data['expansion'],
            'acronym': data['acronym'],
        }


def get_loss(start, start_logits, end, end_logits):
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_logits, start)
    end_loss = loss_fn(end_logits, end)
    loss = start_loss + end_loss
    return loss


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


class BertAD(nn.Module):
    def __init__(self):
        super(BertAD, self).__init__()
        # self.bert = BertModel.from_pretrained(config.MODEL, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.layer = nn.Linear(768, 2)

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, ids, mask, token_type, start=None, end=None):
        output = self.bert(input_ids=ids,
                           attention_mask=mask,
                           token_type_ids=token_type)
        logits = self.layer(output[0])
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = get_loss(start, start_logits, end, end_logits)

        return loss, start_logits, end_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()

    pred_expansion_ = []
    true_expansion_ = []

    for d in data_loader:
        ids = d['ids']
        mask = d['mask']
        token_type = d['token_type']
        start = d['start']
        end = d['end']

        ids = ids.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        model.zero_grad()
        loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer, barrier=True)

        losses.update(loss.item(), ids.size(0))

        # code copied from the eval function
        start_prob = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_prob = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        jac_ = []
        text = d['text']
        expansion = d['expansion']
        offset = d['offset']
        acronym = d['acronym']
        for px, s in enumerate(text):
            start_idx = np.argmax(start_prob[px, :])
            end_idx = np.argmax(end_prob[px, :])

            js, exp = evaluate_jaccard(s, expansion[px], acronym[px], offset[px], start_idx, end_idx)
            jac_.append(js)
            pred_expansion_.append(exp)
            true_expansion_.append(expansion[px])

    pred_expansion_ = [config.A2ID[w] for w in pred_expansion_]
    true_expansion_ = [config.A2ID[w] for w in true_expansion_]

    f1 = f1_score(true_expansion_, pred_expansion_, average='macro')
    precision = precision_score(true_expansion_, pred_expansion_, average='macro')
    recall = recall_score(true_expansion_, pred_expansion_, average='macro')
    return f1, precision, recall, losses.avg


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def evaluate_jaccard(text, selected_text, acronym, offsets, idx_start, idx_end):
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    candidates = config.DICTIONARY[acronym]
    candidate_jaccards = [jaccard(w.strip(), filtered_output.strip()) for w in candidates]
    idx = np.argmax(candidate_jaccards)

    return candidate_jaccards[idx], candidates[idx]


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jac = AverageMeter()

    pred_expansion_ = []
    true_expansion_ = []

    for d in data_loader:
        ids = d['ids']
        mask = d['mask']
        token_type = d['token_type']
        start = d['start']
        end = d['end']

        text = d['text']
        expansion = d['expansion']
        offset = d['offset']
        acronym = d['acronym']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        start_prob = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_prob = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        jac_ = []

        for px, s in enumerate(text):
            start_idx = np.argmax(start_prob[px, :])
            end_idx = np.argmax(end_prob[px, :])

            js, exp = evaluate_jaccard(s, expansion[px], acronym[px], offset[px], start_idx, end_idx)
            jac_.append(js)
            pred_expansion_.append(exp)
            true_expansion_.append(expansion[px])

        jac.update(np.mean(jac_), len(jac_))
        losses.update(loss.item(), ids.size(0))

    pred_expansion_ = [config.A2ID[w] for w in pred_expansion_]
    true_expansion_ = [config.A2ID[w] for w in true_expansion_]

    f1 = f1_score(true_expansion_, pred_expansion_, average='macro')
    precision = precision_score(true_expansion_, pred_expansion_, average='macro')
    recall = recall_score(true_expansion_, pred_expansion_, average='macro')

    return f1, precision, recall, losses.avg, jac.avg


def run(df_train, df_val, df_test, fold):
    train_dataset = Dataset(
        text=df_train.text.values,
        acronym=df_train.acronym_.values,
        expansion=df_train.expansion.values
    )

    valid_dataset = Dataset(
        text=df_val.text.values,
        acronym=df_val.acronym_.values,
        expansion=df_val.expansion.values,
    )

    test_dataset = Dataset(
        text=df_test.text.values,
        acronym=df_test.acronym_.values,
        expansion=df_test.expansion.values,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    model = BertAD()

    if not config.freeze_layer_count:
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

    print('Starting training....')
    best_valid_f1 = 0
    for epoch in range(config.EPOCHS):
        train_f1, train_precision, train_recall, train_loss = train_fn(train_data_loader, model, optimizer, device)
        valid_f1, valid_precision, valid_recall, valid_loss, valid_jac = eval_fn(valid_data_loader, model, device)

        print(
            f'Fold {fold} | Epoch :{epoch + 1}  | Train loss :{train_loss} | Train precision :{train_precision} | Train recall  :{train_recall} | Train f1 :{train_f1}')

        print(
            f'Fold {fold} | Epoch :{epoch + 1}  | Validation jaccard :{valid_jac}| Validation loss :{valid_loss} | Validation precision :{valid_precision} | Validation recall  :{valid_recall} | Validation f1 :{valid_f1}')

        with open(details_file_path, 'a') as file:
            file.write(
                f'\nFold {fold} | Epoch :{epoch + 1}  | Train loss :{train_loss} | Train precision :{train_precision} | Train recall  :{train_recall} | Train f1 :{train_f1}')
            file.write(
                f'\nFold {fold} | Epoch :{epoch + 1}  | Validation jaccard :{valid_jac} | Validation loss :{valid_loss} | Validation precision :{valid_precision} | Validation recall  :{valid_recall} | Validation f1 :{valid_f1}')
        if fold is None:
            path = config.SAVE_DIR + timestamp + 'model.pth'
            torch.save(model.state_dict(), path)
            # es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, "model.bin"))
        elif valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            path = config.SAVE_DIR + timestamp + 'best_model.pth'
            torch.save(model.state_dict(), path)
        else:
            path = config.SAVE_DIR + timestamp + 'model_' + str(fold) + '.pth'
            torch.save(model.state_dict(), path)

    test_f1, test_precision, test_recall, valid_loss, valid_jac = eval_fn(test_data_loader, model, device)
    print(
        f'Test precision :{test_precision} | Test recall  :{test_recall} | Test f1 :{test_f1}')
    with open(details_file_path, 'a') as file:
        file.write(
            f'\nTest precision :{test_precision} | Test recall  :{test_recall} | Test f1 :{test_f1}')

    # return es.best_score
    return valid_f1


def run_k_fold(fold_id):
    '''
      Perform k-fold cross-validation
    '''
    seed_all()

    df_train = pd.read_csv(config.TRAIN_FILE)
    df_val = pd.read_csv(config.VAL_FILE)

    if config.INCLUDE_EXT == True:
        df_ext = pd.read_csv(config.EXTERNAL_FILE)
        train = pd.concat([df_train, df_ext], ignore_index=True)
        df_train = train

    df_test = df_val

    return run(df_train, df_val, df_test, fold_id)


if __name__ == '__main__':
    device = config.DEVICE
    timestamp = datetime.now().strftime(config.time_format) + '/'
    os.makedirs(config.SAVE_DIR + timestamp, exist_ok=True)
    model_file_path = config.SAVE_DIR + timestamp
    details_file_path = config.SAVE_DIR + timestamp + config.DETAILS_FILE
    with open(details_file_path, 'a') as file:
        file.write('File Name: Baseline for AAAI AD data')
        file.write(f'\nStart time:{timestamp} ')
        file.write(
            f'\nNo. of epochs:  {config.EPOCHS} |  Batch size :{config.TRAIN_BATCH_SIZE} | Kfold :{config.KFOLD} | Max len  :{config.MAX_LEN} | Learning Rate :{config.LEARNING_RATE}| Dropout :{config.DROPOUT} | Freeze layer count :{config.freeze_layer_count} |Includes external data for training :{config.INCLUDE_EXT}\n')

    f0 = run_k_fold(0)

    f = [f0]
    # f = [f0]
    end_timestamp = datetime.now().strftime(config.time_format)
    with open(details_file_path, 'a') as file:
        file.write(
            f'\nNo. of folds:  {len(f)} | Final of F score:  {np.mean(f)}')
        file.write(f'\nEnd time:{end_timestamp}')
    print(f'\nFinal of F score:. {np.mean(f)}')
