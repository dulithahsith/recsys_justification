import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Processing Classes (reused from your original script) ---
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class JustificationProcessor(object): # Simplified as it only needs to read for CNN
    """Processor for the Justification dataset."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "label_data_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "label_data_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "label_data_test.csv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = "1" if eval(line[1]) else "0"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# --- CNN Specific Preprocessing ---
class Vocabulary:
    def __init__(self):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}
        self.n_words = 2

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.idx_to_word[self.n_words] = word
            self.n_words += 1
        return self.word_to_idx[word]

    def __len__(self):
        return self.n_words

def build_vocabulary(examples, tokenizer):
    vocab = Vocabulary()
    for example in examples:
        tokens = tokenizer(example.text_a.lower()) # Lowercase for consistency
        for token in tokens:
            vocab.add_word(token)
    return vocab

def texts_to_sequences(examples, vocab, tokenizer, max_seq_length):
    features = []
    for example in examples:
        tokens = tokenizer(example.text_a.lower()) # Lowercase for consistency
        
        # Truncate
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        # Convert tokens to numerical IDs
        input_ids = [vocab.word_to_idx.get(token, vocab.word_to_idx["<unk>"]) for token in tokens]

        # Pad
        padding_length = max_seq_length - len(input_ids)
        input_ids += [vocab.word_to_idx["<pad>"]] * padding_length
        
        # Label ID
        label_id = 1 if eval(example.label) else 0 # Assuming binary classification (True/False -> 1/0)

        assert len(input_ids) == max_seq_length

        features.append({
            'input_ids': input_ids,
            'label_id': label_id
        })
    return features

# A simple tokenizer (space-based for this example)
def simple_tokenizer(text):
    return text.split()

# --- CNN Model Definition ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Three convolution layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, embedding dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, seq len, embedding dim]
        
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, (seq len - filter_sizes[n] + 1)]
        
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)

# --- Metrics (reused from your original script, simplified) ---
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average='binary') # 'binary' for 0/1 labels
    recall = recall_score(y_true=labels, y_pred=preds, average='binary')
    precision = precision_score(y_true=labels, y_pred=preds, average='binary')
    return {
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
    }

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3, # A common learning rate for Adam with CNNs
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0, # Increased epochs for CNN to converge
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    
    # CNN specific arguments
    parser.add_argument("--embedding_dim", default=100, type=int, help="Dimension of word embeddings.")
    parser.add_argument("--n_filters", default=100, type=int, help="Number of filters per convolution layer.")
    parser.add_argument("--filter_sizes", default="3,4,5", type=str, help="Comma-separated filter sizes (e.g., '3,4,5').")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout probability.")

    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # --- Random Seeds ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.warning("Output directory ({}) already exists and is not empty. This might overwrite previous results.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = JustificationProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) # Should be 2 for "0", "1"

    # --- Prepare Data ---
    train_examples = None
    eval_examples = None
    
    # Build vocabulary from training data
    logger.info("Building vocabulary...")
    train_raw_examples = processor.get_train_examples(args.data_dir)
    vocab = build_vocabulary(train_raw_examples, simple_tokenizer)
    logger.info(f"Vocabulary size: {len(vocab)}")

    if args.do_train:
        train_features = texts_to_sequences(
            train_raw_examples, vocab, simple_tokenizer, args.max_seq_length)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f['label_id'] for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        num_train_optimization_steps = int(len(train_dataloader) * args.num_train_epochs)

    # --- Initialize CNN Model ---
    filter_sizes = [int(s) for s in args.filter_sizes.split(',')]
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        n_filters=args.n_filters,
        filter_sizes=filter_sizes,
        output_dim=num_labels, # 2 for binary classification
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss() # For classification

    global_step = 0
    tr_loss = 0

    if args.do_train:
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                input_ids, label_ids = batch
                input_ids = input_ids.to(device)
                label_ids = label_ids.to(device)

                optimizer.zero_grad()
                
                # Forward pass
                logits = model(input_ids)
                
                # Calculate loss
                loss = criterion(logits, label_ids)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1
            tr_loss = epoch_loss / len(train_dataloader) # Average loss per epoch
            logger.info(f"Epoch {epoch+1} training loss: {tr_loss:.4f}")


    # --- Evaluation ---
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = texts_to_sequences(
            eval_examples, vocab, simple_tokenizer, args.max_seq_length)

        logger.info("***** Running evaluation on Dev Set *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f['label_id'] for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_input_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() # Set model to evaluation mode
        eval_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(eval_dataloader, desc="Evaluating Dev"):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad(): # Disable gradient calculation for evaluation
                logits = model(input_ids)
            
            loss = criterion(logits, label_ids)
            eval_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        eval_loss = eval_loss / len(eval_dataloader)
        result = compute_metrics(all_preds, all_labels)
        result['eval_loss'] = eval_loss
        
        output_eval_file = os.path.join(args.output_dir, "dev_results_cnn.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Dev Eval results (CNN) *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {result[key]:.4f}")
                writer.write(f"{key} = {result[key]:.4f}\n")
        
        # --- Test Evaluation (similar to dev eval) ---
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = texts_to_sequences(
            test_examples, vocab, simple_tokenizer, args.max_seq_length)

        logger.info("***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f['input_ids'] for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f['label_id'] for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval() # Set model to evaluation mode
        test_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(test_dataloader, desc="Evaluating Test"):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = model(input_ids)
            
            loss = criterion(logits, label_ids)
            test_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        test_loss = test_loss / len(test_dataloader)
        result = compute_metrics(all_preds, all_labels)
        result['test_loss'] = test_loss
        
        output_test_file = os.path.join(args.output_dir, "test_results_cnn.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results (CNN) *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {result[key]:.4f}")
                writer.write(f"{key} = {result[key]:.4f}\n")


if __name__ == "__main__":
    main()