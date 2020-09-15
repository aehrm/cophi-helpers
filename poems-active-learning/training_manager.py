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

        # zero indicates "other" class
        self.classify_map = {"Natur": 1, "Liebe": 2}
        self.inverse_classify_map = {1: "Natur", 2: "Liebe", 0: "O"}

        self.classification_output = None
        self.status_object = {}

    def is_training(self):
        return self.future is not None and not self.future.done()

    def get_status_object(self):
        return self.status_object

    def start_training(self):
        self.future = self.pool.submit(self._train_and_classify)
        return self.future

    def get_label_batch(self, k=20, classes=set(["Natur", "Liebe"])):
        def get_content(filename):
            with open(os.path.join(self.document_dir, filename)) as f:
                lines = [x.strip() for x in f]
                relevant_lines = list(map(lambda x: x[0], itertools.takewhile(lambda x: x[1] < 512,
                    zip(lines, itertools.accumulate(map(lambda l: len(l.split()), lines))))))
                if len(relevant_lines) < len(lines):
                    return '\n'.join(relevant_lines) + '\n\n[...]'
                else:
                    return '\n'.join(relevant_lines)

        df = self.classification_output.merge(self.labeled_documents, how='outer', on='filename')
        docs = df.where(df['label'].isnull()) \
            .where(df['predicted_label'].apply(lambda x: x in classes)) \
            .drop(columns=['label', 'label_date']) \
            .dropna()

        if len(docs) >= k:
            docs = docs.sample(n=k, replace=False)
            docs['content'] = docs['filename'].apply(get_content)
        else:
            unlabeled = df.where(df['label'].isnull()) \
                .where(df['predicted_label'].apply(lambda x: x not in classes)) \
                .drop(columns=['label', 'label_date'])
            docs = docs.append(unlabeled.sample(n=k-len(docs), replace=False))

        return docs

    def add_labeled(self, labeled_data):
        """ labeled_data is a (filename, label) data frame """
        df = labeled_data.copy()[['filename', 'label']]
        df['label_date'] = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        self.labeled_documents = self.labeled_documents.append(df)
        self.labeled_documents.to_csv(self.labeled_documents_file, sep="\t", index=False)

        return self.labeled_documents

    def _get_sequence(self, filename):
        with open(os.path.join(self.document_dir, filename)) as f:
            text = list(more_itertools.flatten([x.strip() for x in f]))
        encoded = self.tokenizer.encode_plus(text, padding="max_length", truncation=True)
        return encoded

    def _train_and_classify(self):
        def transform_label(x):
            if x in self.classify_map.keys():
                return self.classify_map[x]
            else:
                return 0

        self.status_object = {'stage': 'init', 'startdate': datetime.datetime.now().astimezone().replace(microsecond=0).isoformat() }

        X = self.labeled_documents["filename"]
        Y = self.labeled_documents["label"].apply(transform_label)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                          test_size=0.20,
                                                          stratify=Y)

        # tokenize
        mask = []
        input_ids = []
        for file in X_train:
            encoded = self._get_sequence(file)
            input_ids.append(encoded["input_ids"])
            mask.append(encoded["attention_mask"])
        mask_train = torch.tensor(mask)
        input_ids_train = torch.tensor(input_ids)

        mask = []
        input_ids = []
        for file in X_val:
            encoded = self._get_sequence(file)
            input_ids.append(encoded["input_ids"])
            mask.append(encoded["attention_mask"])
        mask_val = torch.tensor(mask)
        input_ids_val = torch.tensor(input_ids)

        # prepare dataloader
        train_dataset = TensorDataset(input_ids_train, mask_train, torch.tensor(np.array([x for x in Y_train])))
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)

        validation_dataset = TensorDataset(input_ids_val, mask_val, torch.tensor(np.array([x for x in Y_val])))
        validation_sampler = SequentialSampler(validation_dataset)
        validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=8)

        # init model
        model = BertForSequenceClassification.from_pretrained("bert-base-german-dbmdz-cased",
                                                              num_labels=len(self.inverse_classify_map),
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
        best_val_loss = math.inf
        best_classifier = None
        t = tqdm(total=len(train_dataset)*epochs, ncols=150)
        for epoch_i in range(0, epochs):
            model.train()
            torch.set_grad_enabled(True)
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                eval_loss = outputs[0]
                eval_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss = eval_loss.detach().cpu().numpy() / train_dataloader.batch_size
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
            eval_loss = 0
            for step, batch in enumerate(validation_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                eval_loss += outputs[0].detach().cpu().numpy()

            # store best model w.r.t. the validation set
            if best_classifier is None or eval_loss < best_val_loss:
                best_val_loss = eval_loss
                best_classifier = model.classifier.state_dict()

            self.status_object['train_info']['val_loss'].append(eval_loss / len(validation_dataset))
            t.set_postfix({'train': f"{self.status_object['train_info']['train_loss'][-1]:.3e}",
                           'val': f"{self.status_object['train_info']['val_loss'][-1]:.3e}"})
        t.close()

        model.classifier.load_state_dict(best_classifier)
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
            b_input_mask =batch[1].to(device)
            b_filenames = batch[2]
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            preds = np.argmax(outputs[0].detach().cpu().numpy(), axis=-1)
            for i, file in enumerate(b_filenames):
                classification[file] = self.inverse_classify_map[preds[i]]
            t.update(len(b_filenames))
            self.status_object['predict_info']['labeled'] += len(b_filenames)
        t.close()

        # print F1 score over all labeled documents
        pred = self.labeled_documents['filename'].apply(lambda x: classification[x])
        label = self.labeled_documents['label'].apply(
            lambda x: self.inverse_classify_map[transform_label(x)])  # normalize labels
        self.status_object['result'] = classification_report(label, pred, output_dict=True)
        print(classification_report(label, pred))

        # store classification
        self.classification_output = pandas.DataFrame(list(classification.items()),
                                                      columns=['filename', 'predicted_label'])
        self.status_object['stage'] = 'done'
        self.status_object['enddate'] = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        print('Training done')


class DocDataset(Dataset):

    def __init__(self, documents, tokenize_method):
        self.documents = documents
        self.tokenize_method = tokenize_method

    def __getitem__(self, item):
        encoded = self.tokenize_method(self.documents[item])
        return [torch.tensor(encoded['input_ids']), torch.tensor(encoded['attention_mask']), self.documents[item]]

    def __len__(self):
        return len(self.documents)
