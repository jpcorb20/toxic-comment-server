# -*- coding: utf-8 -*-
"""
Original Colab file is located at
    https://colab.research.google.com/github/rap12391/transformers_multilabel_toxic/blob/master/toxic_multilabel.ipynb

Source: https://towardsdatascience.com/transformers-for-multilabel-classification-71a1a0daf5e1
"""

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

df = pd.read_csv('data/train.csv')

cols = df.columns
label_cols = list(cols[2:])

df = df.sample(frac=1).reset_index(drop=True)

df['one_hot_labels'] = list(df[label_cols].values)

labels = list(df.one_hot_labels.values)
comments = list(df.comment_text.values)


def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.cuda()
    return tokenizer, model


tokenizer, model = load_model()

encodings = tokenizer.batch_encode_plus(comments, max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)

input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']

train_inputs, validation_inputs, \
train_labels, validation_labels, \
train_masks, validation_masks = train_test_split(input_ids, labels, attention_masks,
                                                 random_state=RDN_NUM, test_size=0.10, stratify=labels)


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


train_dataloader = process_training_data(train_inputs, train_labels, train_masks)
validation_dataloader = process_validation_data(validation_inputs, validation_labels, validation_masks)


def config_optimizer():
    # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    return AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=True)


def train(train_dataloader):
    optimizer = config_optimizer()

    train_loss_set = []
    for _ in trange(N_EPOCHS, desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, labels = batch

            optimizer.zero_grad()

            # Forward pass for multilabel classification
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits.view(-1, NUM_LABELS), labels.type_as(logits).view(-1, NUM_LABELS))
            train_loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        model.eval()

        logit_preds, true_labels, pred_labels = [], [], []

        for i, batch in enumerate(validation_dataloader):

            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                pred_label = torch.sigmoid(outputs[0])

            logit_preds.append(outputs[0].detach().cpu().numpy())
            true_labels.append(labels.to('cpu').numpy())
            pred_labels.append(pred_label.to('cpu').numpy())

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl > threshold for pl in pred_labels]
        true_bools = [tl == 1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools, pred_bools, average='macro')

        print('Macro F1 on validation: ', val_f1_accuracy)

    torch.save(model.state_dict(), 'model')


test_df = pd.read_csv('data/test.csv')
test_labels_df = pd.read_csv('data/test_labels.csv')
test_df = test_df.merge(test_labels_df, on='id', how='left')
test_label_cols = list(test_df.columns[2:])

test_df = test_df[~test_df[test_label_cols].eq(-1).any(axis=1)] #remove irrelevant rows/comments with -1 values
test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

test_labels = list(test_df.one_hot_labels.values)
test_comments = list(test_df.comment_text.values)

test_encodings = tokenizer.batch_encode_plus(test_comments, max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True)
test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']

# Make tensors out of data
test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_attention_masks)
# Create test dataloader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
# Save test dataloader
torch.save(test_dataloader, 'data/test_data_loader')

# Test

# Put model in evaluation mode to evaluate loss on the validation set
model.eval()

#track variables
logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

# Predict
for i, batch in enumerate(test_dataloader):
    batch = tuple(t.to(DEVICE) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        # Forward pass
        outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)

        b_logit_pred = b_logit_pred.detach().cpu().numpy()
        pred_label = pred_label.to('cpu').numpy()
        b_labels = b_labels.to('cpu').numpy()

    tokenized_texts.append(b_input_ids)
    logit_preds.append(b_logit_pred)
    true_labels.append(b_labels)
    pred_labels.append(pred_label)

# Flatten outputs
tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]
true_labels = [item for sublist in true_labels for item in sublist]
# Converting flattened binary values to boolean values
true_bools = [tl == 1 for tl in true_labels]

pred_bools = [pl > 0.50 for pl in pred_labels] #boolean output after thresholding

# Print and save classification report
print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='micro'))
print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
clf_report = classification_report(true_bools,pred_bools,target_names=test_label_cols)
pickle.dump(clf_report, open('classification_report.txt','wb')) #save report
print(clf_report)
