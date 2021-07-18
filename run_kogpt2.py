


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


# 전처리
with open("dataset_original_json/texts.json", "r", encoding="UTF-8") as fp:
    document_gpt2 = json.load(fp)
with open("dataset_augmented_json/texts_augmented.json", "r", encoding="UTF-8") as fp:
    document_gpt2 += json.load(fp)
document_gpt2 = [
    vocab.bos_token + " " + str(s) + " " + vocab.eos_token for s in document_gpt2
]
tokenized_texts = []  # 입력값 저장을 위한 리스트
attention_masks = []  # attention mask 저장을 위한 리스트
MAX_LEN = 64  # sentence의 최대 길이
for text in document_gpt2:
    tokenized = tokenizer(text)  # text 토크나이징
    token_to_idx = [vocab.token_to_idx[s] for s in tokenized]  # token값을 index로 변경

    # attention mask 추가
    attention = (
        [0] * (MAX_LEN - len(token_to_idx)) + [1] * len(token_to_idx)
        if len(token_to_idx) < MAX_LEN
        else [1] * MAX_LEN
    )
    # token에 padding index 붙이기(KoGPT2는 padding을 왼쪽에 붙여야 함)
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

# dataloader 설정


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
            embedding_size, num_labels, bias=False  # 중요한 부분!
        )  # Classification을 위해 최종단에 선형 레이어 추가
        self.num_labels = num_labels  # label class 갯수
        self.softmax = nn.Softmax()  # 모델 성능을 높이기 위한 임시조치

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]  # GPT2Model의 출력값 크기 (batch_size, sequence_length, embedding_size)
        logits = self.score(
            hidden_states
        )  # Linear layer 출력 (batch_size, sequence_length, num_labels)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]

        sequence_length = -1

        pooled_logits = logits[
            range(batch_size), sequence_length
        ]  # last sequence token 값만 사용 (bs, num_labels)
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
)  # 이거인가?


# 옵티마이저 설정
optimizer = AdamW(
    gpt2_classification_model.parameters(),
    lr=5e-5,  # 학습률
    eps=1e-8,  # 0으로 나누는 것을 방지하기 위한 epsilon 값
)

# 에폭수
epochs = 4

# 총 훈련 스텝
total_steps = len(train_dataloader) * epochs

# 정확도 계산 함수


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    import random
    import numpy as np
    import time
    import datetime

    # 사용할 device 설정
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.is_available())
    # 재현을 위해 랜덤시드 고정
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    gpt2_classification_model = nn.DataParallel(gpt2_classification_model)
    # 그래디언트 초기화
    gpt2_classification_model.zero_grad()
    gpt2_classification_model.to(device)
    # 에폭만큼 반복
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # 시작 시간 설정
        t0 = time.time()

        # Loss 초기화
        total_loss = 0

        # 훈련모드로 변경
        gpt2_classification_model.train()

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_dataloader):
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.  Loss: {:}  Elapsed: {:}.".format(
                        step, len(train_dataloader), loss.mean().item(), elapsed
                    )
                )
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch
            # Forward 수행
            outputs = gpt2_classification_model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels
            )

            # 로스 구함
            loss = outputs[0]

            # 총 로스 계산
            total_loss += loss.mean().item()

            # Backward 수행으로 그래디언트 계산
            loss.mean().backward()

            # 그래디언트를 통해 가중치 파라미터 업데이트
            optimizer.step()

            # 그래디언트 초기화
            gpt2_classification_model.zero_grad()

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        # 시작 시간 설정
        t0 = time.time()

        # 평가모드로 변경
        gpt2_classification_model.eval()

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for batch in validation_dataloader:
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # 그래디언트 계산 안함
            with torch.no_grad():
                # Forward 수행
                outputs = gpt2_classification_model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

            # 출력 설정
            logits = outputs[1]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    torch.save(gpt2_classification_model.state_dict(), "./대전과학고")

