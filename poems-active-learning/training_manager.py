import collections
import concurrent.futures
import datetime
import math
import os.path
from os import listdir

import itertools
import more_itertools
import numpy as np
import pandas
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler, Dataset
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from model import BertForBRSequenceClassification


class TrainingManager:

    def __init__(self, document_dir='./poems', labeled_documents_file='./labeled_documents.tsv'):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-german-dbmdz-cased", padding="max_length",
                                                       truncation=True)
        self.future = None
        self.labeled_documents_file = labeled_documents_file
        self.labeled_documents = pandas.read_csv(labeled_documents_file, sep="\t", index_col=False)
        self.document_dir = document_dir
        self.documents = [f for f in listdir(document_dir) if os.path.isfile(os.path.join(document_dir, f))]
        self.classes = ['Natur', 'Liebe']

        self.classification_output = None
        self.status_object = {}

    def is_training(self):
        return self.future is not None and not self.future.done()

    def get_status_object(self):
        return self.status_object

    def start_training(self):
        self.future = self.pool.submit(self._train_and_classify)
        return self.future

    def get_label_batch(self, k=20):
        def get_content(filename):
            with open(os.path.join(self.document_dir, filename)) as f:
                lines = [x.strip() for x in f]
                relevant_lines = list(map(lambda x: x[0], itertools.takewhile(lambda x: x[1] < 512,
                                                                              zip(lines, itertools.accumulate(
                                                                                  map(lambda l: len(l.split()),
                                                                                      lines))))))
                if len(relevant_lines) < len(lines):
                    return '\n'.join(relevant_lines) + '\n\n[...]'
                else:
                    return '\n'.join(relevant_lines)

        df = self.classification_output.merge(self.labeled_documents, how='outer', on='filename')
        docs = df.where(df['labels'].isnull()) \
            .where(df['predicted_labels'].apply(lambda x: len(x & set(self.classes)) > 0)) \
            .drop(columns=['labels', 'label_date']) \
            .dropna()

        if len(docs) >= k:
            docs = docs.sample(n=k, replace=False)
            docs['content'] = docs['filename'].apply(get_content)
        else:
            unlabeled = df.where(df['labels'].isnull()) \
                .where(df['predicted_labels'].apply(lambda x: len(x) == 0)) \
                .drop(columns=['labels', 'label_date']) \
                .dropna()
            docs = docs.append(unlabeled.sample(n=k-len(docs), replace=False))

        docs['predicted_labels'] = docs['predicted_labels'].apply(list)
        return docs

    def add_labeled(self, labeled_data):
        """ labeled_data is a (filename, labels) data frame """
        df = labeled_data.copy()[['filename', 'labels']]
        df['label_date'] = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        df['labels'] = df['labels'].apply(lambda x: x if type(x) == str else ','.join(x))
        self.labeled_documents = self.labeled_documents.append(df)
        self.labeled_documents.to_csv(self.labeled_documents_file, sep="\t", index=False)

        return self.labeled_documents

    def _get_sequence(self, filename):
        with open(os.path.join(self.document_dir, filename)) as f:
            text = list(more_itertools.flatten([x.strip() for x in f]))
        encoded = self.tokenizer.encode_plus(text, padding="max_length", truncation=True)
        return encoded

    def _train_and_classify(self):
        def multihot(classes):
            return torch.BoolTensor(list(map(lambda c: c in classes, self.classes)))

        self.status_object = {'stage': 'init',
                              'startdate': datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()}

        X = self.labeled_documents['filename']
        Y = self.labeled_documents['labels'].apply(lambda x: multihot(set(x.split(','))))
        # TODO how to stratifiy?
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20)

        # prepare dataloader
        train_dataset = DocDataset(X_train.to_list(), tokenize_method=self._get_sequence, labels=Y_train.to_list(), cache=True)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)

        validation_dataset = DocDataset(X_val.to_list(), tokenize_method=self._get_sequence, labels=Y_val.to_list(), cache=True)
        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=8)

        # init model
        model = BertForBRSequenceClassification.from_pretrained("bert-base-german-dbmdz-cased",
                                                              num_labels=len(self.classes),
                                                              attention_probs_dropout_prob=0.1,
                                                              hidden_dropout_prob=0.1)

        epochs = 15
        total_steps = len(train_dataloader) * epochs
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        device = "cuda"
        model.to(device)

        # training loop
        self.status_object['stage'] = 'train'
        self.status_object['train_info'] = {'batches': len(train_dataloader), 'epochs': epochs,
                                            'train_data': len(train_dataset), 'val_data': len(validation_dataset),
                                            'train_loss': [], 'val_loss': []}
        print(
            f"Training classifier on {len(train_dataset)} datapoints ({len(validation_dataset)} validation datapoints)")
        best_val_losses = [math.inf] * len(self.classes)
        best_classifiers = [None] * len(self.classes)
        t = tqdm(total=len(train_dataset) * epochs, ncols=150)
        for epoch_i in range(0, epochs):
            model.train()
            torch.set_grad_enabled(True)
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[3].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0].sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss = loss.detach().cpu().numpy() / b_labels.shape[0]
                self.status_object['train_info']['train_loss'].append(train_loss)
                if len(self.status_object['train_info']['val_loss']) > 0:

                    t.set_postfix({'epoch': epoch_i,
                                   'train': f"{self.status_object['train_info']['train_loss'][-1]:.3e}",
                                   'val': f"{self.status_object['train_info']['val_loss'][-1]:.3e}"})
                else:
                    t.set_postfix({'epoch': epoch_i,
                                   'train': f"{self.status_object['train_info']['train_loss'][-1]:.3e}"})
                t.update(b_labels.size()[0])

            # evaluate against validation set
            model.eval()
            model.zero_grad()
            torch.set_grad_enabled(False)
            eval_losses = [0] * len(self.classes)
            for step, batch in enumerate(validation_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[3].to(device)
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                losses = outputs[0].detach().cpu().numpy()
                for i in range(len(self.classes)):
                    eval_losses[i] += losses[i]

            # store best model w.r.t. the validation set
            for i in range(len(self.classes)):
                if best_classifiers[i] is None or eval_losses[i] < best_val_losses[i]:
                    best_val_losses[i] = eval_losses[i]
                    best_classifiers[i] = model.classifiers[i].state_dict()

            self.status_object['train_info']['val_loss'].append(sum(eval_losses) / len(validation_dataset))
            t.set_postfix({'epoch': epoch_i,
                           'train': f"{self.status_object['train_info']['train_loss'][-1]:.3e}",
                           'val': f"{self.status_object['train_info']['val_loss'][-1]:.3e}"})
        t.close()

        # restore best model
        for i in range(len(self.classes)):
            model.classifiers[i].load_state_dict(best_classifiers[i])
        model.eval()
        model.zero_grad()
        torch.set_grad_enabled(False)

        self.status_object['stage'] = 'predict'
        self.status_object['predict_info'] = {'documents': len(self.documents), 'labeled': 0}
        print(f"Classifying all {len(self.documents)} documents")
        # classify on entire dataset
        classify_dataset = DocDataset(self.documents, tokenize_method=self._get_sequence)
        classify_sampler = RandomSampler(classify_dataset)
        classify_dataloader = DataLoader(classify_dataset, sampler=classify_sampler, batch_size=8)
        classification = dict()

        t = tqdm(total=len(classify_dataset))
        for step, batch in enumerate(classify_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_filenames = batch[2]
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            preds = np.argmax(outputs[0].detach().cpu().numpy(), axis=2)
            for i, file in enumerate(b_filenames):
                classification[file] = set([self.classes[j] for j in range(len(self.classes)) if preds[i][j]])
            t.update(len(b_filenames))
            self.status_object['predict_info']['labeled'] += len(b_filenames)
        t.close()

        # for each class, print F1 score over all labeled documents
        self.status_object['result'] = dict()
        for cl in self.classes:
            pred = self.labeled_documents['filename'].apply(lambda x: cl if cl in classification[x] else 'Non-' + cl)
            label = self.labeled_documents['labels'].apply(lambda x: cl if cl in x.split(',') else 'Non-' + cl)
            self.status_object['result'][cl] = classification_report(label, pred, output_dict=True)
            print(classification_report(label, pred))

        # store classification
        self.classification_output = pandas.DataFrame(list(classification.items()),
                                                      columns=['filename', 'predicted_labels'])
        self.status_object['stage'] = 'done'
        self.status_object['enddate'] = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        print('Training done')


class DocDataset(Dataset):

    def __init__(self, documents, tokenize_method, labels=None, cache=False):
        self.documents = documents
        self.tokenize_method = tokenize_method
        self.labels = labels
        self.cache = None
        if cache:
            self.cache = [None] * len(documents)

    def __getitem__(self, item):
        if self.cache is not None and self.cache[item] is not None:
            encoded = self.cache[item]
        else:
            encoded = self.tokenize_method(self.documents[item])
            if self.cache is not None:
                self.cache[item] = encoded

        if self.labels is not None:
            return [torch.tensor(encoded['input_ids']), torch.tensor(encoded['attention_mask']), self.documents[item], self.labels[item]]
        else:
            return [torch.tensor(encoded['input_ids']), torch.tensor(encoded['attention_mask']), self.documents[item]]

    def __len__(self):
        return len(self.documents)
