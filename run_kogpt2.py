


# coding=utf-8
# Copyright 2020 SKT AIX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gluonnlp as nlp
import torch
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from gluonnlp.data import SentencepieceTokenizer
from sklearn.model_selection import train_test_split
import json

pytorch_kogpt2 = {
    "url": "https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params",
    "fname": "pytorch_kogpt2_676e9bcfa7.params",
    "chksum": "676e9bcfa7",
}
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "output_past": True,
}


def remove_module(d):
    ret = {}
    for k, v in d.items():
        if k.startswith("module."):
            ret[k[7:]] = v
        else:
            return d

    return ret


def get_kogpt2_model(model_file, vocab_file, ctx="cpu"):
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    d = torch.load(model_file)
    d = remove_module(d)
    kogpt2model.load_state_dict(d)
    device = torch.device(ctx)
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file,
        mask_token=None,
        sep_token=None,
        cls_token=None,
        unknown_token="<unk>",
        padding_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return kogpt2model, vocab_b_obj


tok_path = (
    "kogpt2_files/kogpt2_news_wiki_ko_cased_818bfa919d.spiece"
)
model, vocab = get_kogpt2_model(
    "kogpt2_files/pytorch_kogpt2_676e9bcfa7.params",
    "kogpt2_files/kogpt2_news_wiki_ko_cased_818bfa919d.spiece",
    ctx="cpu",
)
tokenizer = SentencepieceTokenizer(tok_path, 0, 0)


# ?????????
with open("dataset_original_json/texts.json", "r", encoding="UTF-8") as fp:
    document_gpt2 = json.load(fp)
with open("dataset_augmented_json/texts_augmented.json", "r", encoding="UTF-8") as fp:
    document_gpt2 += json.load(fp)
document_gpt2 = [
    vocab.bos_token + " " + str(s) + " " + vocab.eos_token for s in document_gpt2
]
tokenized_texts = []  # ????????? ????????? ?????? ?????????
attention_masks = []  # attention mask ????????? ?????? ?????????
MAX_LEN = 64  # sentence??? ?????? ??????
for text in document_gpt2:
    tokenized = tokenizer(text)  # text ???????????????
    token_to_idx = [vocab.token_to_idx[s] for s in tokenized]  # token?????? index??? ??????

    # attention mask ??????
    attention = (
        [0] * (MAX_LEN - len(token_to_idx)) + [1] * len(token_to_idx)
        if len(token_to_idx) < MAX_LEN
        else [1] * MAX_LEN
    )
    # token??? padding index ?????????(KoGPT2??? padding??? ????????? ????????? ???)
    token_to_idx = (
        [vocab.token_to_idx[vocab.padding_token]] * (MAX_LEN - len(token_to_idx))
        + token_to_idx
        if len(token_to_idx) < MAX_LEN
        else token_to_idx[0:MAX_LEN]
    )
    tokenized_texts.append(token_to_idx)
    attention_masks.append(attention)
with open("dataset_original_json/labels.json", "r", encoding="UTF-8") as fp:
    labels = json.load(fp)
with open("dataset_augmented_json/labels_augmented.json", "r", encoding="UTf-8") as fp:
    labels += json.load(fp)
(
    trainplusvalidation_inputs,
    test_inputs,
    trainplusvalidation_labels,
    test_labels,
) = train_test_split(
    tokenized_texts, labels, random_state=42, test_size=0.1  # needs to be fixed
)
trainplusvalidation_masks, test_attention_masks, _, _ = train_test_split(
    attention_masks, tokenized_texts, random_state=42, test_size=0.1
)


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    # needs to be fixed
    trainplusvalidation_inputs,
    trainplusvalidation_labels,
    random_state=42,
    test_size=0.1,
)
train_masks, validation_masks, _, _ = train_test_split(
    trainplusvalidation_masks,
    trainplusvalidation_inputs,
    random_state=42,
    test_size=0.1,
)

# dataloader ??????


class NaverMovieDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids)
        self.attention_masks = torch.tensor(attention_masks)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


BATCH_SIZE = 32
train_data = NaverMovieDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

validation_data = NaverMovieDataset(
    validation_inputs, validation_masks, validation_labels
)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = NaverMovieDataset(test_inputs, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


class GPT2ForSequenceClassification(nn.Module):
    def __init__(self, gpt2, embedding_size, num_labels):
        super().__init__()
        self.gpt2 = gpt2  # huggingface transformer GPT2Model
        self.score = nn.Linear(
            embedding_size, num_labels, bias=False  # ????????? ??????!
        )  # Classification??? ?????? ???????????? ?????? ????????? ??????
        self.num_labels = num_labels  # label class ??????
        self.softmax = nn.Softmax()  # ?????? ????????? ????????? ?????? ????????????

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]  # GPT2Model??? ????????? ?????? (batch_size, sequence_length, embedding_size)
        logits = self.score(
            hidden_states
        )  # Linear layer ?????? (batch_size, sequence_length, num_labels)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]

        sequence_length = -1

        pooled_logits = logits[
            range(batch_size), sequence_length
        ]  # last sequence token ?????? ?????? (bs, num_labels)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    self.softmax(pooled_logits.view(-1, self.num_labels)),
                    labels.view(-1),
                )

        return (
            loss,
            pooled_logits,
        )  # loss: CrossEntropyLoss, pooled_logits: (bs, num_labels)


gpt2_classification_model = GPT2ForSequenceClassification(
    model.transformer, 768, 7
)  # ?????????????


# ??????????????? ??????
optimizer = AdamW(
    gpt2_classification_model.parameters(),
    lr=5e-5,  # ?????????
    eps=1e-8,  # 0?????? ????????? ?????? ???????????? ?????? epsilon ???
)

# ?????????
epochs = 4

# ??? ?????? ??????
total_steps = len(train_dataloader) * epochs

# ????????? ?????? ??????


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# ?????? ?????? ??????
def format_time(elapsed):
    # ?????????
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss?????? ?????? ??????
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    import random
    import numpy as np
    import time
    import datetime

    # ????????? device ??????
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.is_available())
    # ????????? ?????? ???????????? ??????
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    gpt2_classification_model = nn.DataParallel(gpt2_classification_model)
    # ??????????????? ?????????
    gpt2_classification_model.zero_grad()
    gpt2_classification_model.to(device)
    # ???????????? ??????
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # ?????? ?????? ??????
        t0 = time.time()

        # Loss ?????????
        total_loss = 0

        # ??????????????? ??????
        gpt2_classification_model.train()

        # ????????????????????? ???????????? ???????????? ?????????
        for step, batch in enumerate(train_dataloader):
            # ?????? ?????? ??????
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.  Loss: {:}  Elapsed: {:}.".format(
                        step, len(train_dataloader), loss.mean().item(), elapsed
                    )
                )
            # ????????? GPU??? ??????
            batch = tuple(t.to(device) for t in batch)

            # ???????????? ????????? ??????
            b_input_ids, b_input_mask, b_labels = batch
            # Forward ??????
            outputs = gpt2_classification_model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels
            )

            # ?????? ??????
            loss = outputs[0]

            # ??? ?????? ??????
            total_loss += loss.mean().item()

            # Backward ???????????? ??????????????? ??????
            loss.mean().backward()

            # ?????????????????? ?????? ????????? ???????????? ????????????
            optimizer.step()

            # ??????????????? ?????????
            gpt2_classification_model.zero_grad()

        # ?????? ?????? ??????
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        # ?????? ?????? ??????
        t0 = time.time()

        # ??????????????? ??????
        gpt2_classification_model.eval()

        # ?????? ?????????
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # ????????????????????? ???????????? ???????????? ?????????
        for batch in validation_dataloader:
            # ????????? GPU??? ??????
            batch = tuple(t.to(device) for t in batch)

            # ???????????? ????????? ??????
            b_input_ids, b_input_mask, b_labels = batch

            # ??????????????? ?????? ??????
            with torch.no_grad():
                # Forward ??????
                outputs = gpt2_classification_model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

            # ?????? ??????
            logits = outputs[1]

            # CPU??? ????????? ??????
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # ?????? ????????? ????????? ???????????? ????????? ??????
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    torch.save(gpt2_classification_model.state_dict(), "./???????????????")

