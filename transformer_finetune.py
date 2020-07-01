# -*- coding: utf-8 -*-
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'distilroberta-base'
N_EPOCHS = 3
BATCH_SIZE = 32
MAX_LENGTH = 128
RDN_NUM = 42
LEARNING_RATE = 2e-5
NUM_LABELS = 6
LOGIT_THRESHOLD = 0.5

tokenizer, model = None, None


def flatten_double_nested_list(list_obj):
    """
    Flatten nested list on two levels.
    :param list_obj: List[list].
    :return: list.
    """
    return [e for l in list_obj for e in l]


def load_model():
    """
    Load huggingface's model and tokenizer.
    :return: tokenizer and model.
    """

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.cuda()

    return tokenizer, model


def load_train_val_data(split_ratio=0.20):
    df = pd.read_csv('data/train.csv')

    cols = df.columns
    label_cols = list(cols[2:])

    df['one_hot_labels'] = list(df[label_cols].values)

    labels = list(df.one_hot_labels.values)
    comments = list(df.comment_text.values)

    encodings = tokenizer.batch_encode_plus(comments, max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    train_inputs, validation_inputs, \
    train_labels, validation_labels, \
    train_masks, validation_masks = train_test_split(input_ids, labels, attention_masks,
                                                     random_state=RDN_NUM, test_size=split_ratio)

    return (train_inputs, train_labels, train_masks),\
           (validation_inputs, validation_labels, validation_masks)


def process_training_data(train_inputs, train_labels, train_masks):
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    torch.save(train_dataloader, 'data/train_data_loader')

    return train_dataloader


def process_validation_data(validation_inputs, validation_labels, validation_masks):
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    torch.save(validation_dataloader, 'data/validation_data_loader')

    return validation_dataloader


def config_optimizer():
    """
    Configure optimer.
    :return: optimizer (transformers.AdamW).
    """
    global model

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    return AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=True)


def train(train_dataloader, validation_dataloader):
    global model, DEVICE, N_EPOCHS, NUM_LABELS

    optimizer = config_optimizer()
    loss_func = BCEWithLogitsLoss()

    train_loss_set = []
    for _ in trange(N_EPOCHS, desc="Epoch"):

        model.train()

        # Train step
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, labels = batch

            optimizer.zero_grad()

            # Forward pass for multilabel classification
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = outputs[0]

            loss = loss_func(logits.view(-1, NUM_LABELS), labels.type_as(logits).view(-1, NUM_LABELS))
            train_loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Eval step
        evaluation(validation_dataloader)

    torch.save(model.state_dict(), 'model')


def load_test_data():
    test_df = pd.read_csv('data/test.csv')
    test_labels_df = pd.read_csv('data/test_labels.csv')
    test_df = test_df.join(test_labels_df, on='id')

    test_label_cols = list(test_df.columns[2:])

    # Remove item with all -1 which are not really test samples.
    test_df = test_df[~test_df[test_label_cols].eq(-1).any(axis=1)]

    test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

    test_encodings = tokenizer.batch_encode_plus(list(test_df.comment_text.values),
                                                 max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)

    return (test_encodings['input_ids'], list(test_df.one_hot_labels.values), test_encodings['attention_mask']),\
        test_label_cols


def process_test_data(test_input_ids, test_labels, test_attention_masks):
    global BATCH_SIZE

    test_inputs = torch.tensor(test_input_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    torch.save(test_dataloader, 'data/test_data_loader')

    return test_dataloader


def evaluation(test_dataloader):
    global model, DEVICE

    model.eval()

    logit_preds, true_labels, pred_labels = list(), list(), list()
    for i, batch in enumerate(test_dataloader):

        batch = tuple(t.to(DEVICE) for t in batch)

        input_ids, input_mask, labels = batch

        with torch.no_grad():
            outs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            pred_label = torch.sigmoid(outs[0])

        logit_preds.append(outs[0].detach().cpu().numpy())
        true_labels.append(labels.to('cpu').numpy())
        pred_labels.append(pred_label.to('cpu').numpy())

    pred_labels = flatten_double_nested_list(pred_labels)
    true_labels = flatten_double_nested_list(true_labels)

    true_bools = [t == 1 for t in true_labels]
    pred_bools = [p > LOGIT_THRESHOLD for p in pred_labels]

    print('Test Macro F1: ', f1_score(true_bools, pred_bools, average='macro'))

    return true_bools, pred_bools


def compute_final_evaluation(test_dataloader, test_labels):
    true_bools, pred_bools = evaluation(test_dataloader)

    clf_report = classification_report(true_bools, pred_bools, target_names=test_labels)

    with open('classification_report.txt', 'wb') as fp:
        pickle.dump(clf_report, fp)


if __name__ == "__main__":
    tokenizer, model = load_model()

    train_data, val_data = load_train_val_data()

    train_dataloader = process_training_data(*train_data)
    validation_dataloader = process_validation_data(*val_data)
    train(train_dataloader, validation_dataloader)

    test_data, test_labels = load_test_data()
    test_dataloader = process_test_data(*test_data)
    compute_final_evaluation(test_dataloader, test_labels)
