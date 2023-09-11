import pandas as pd
import numpy as np
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup, logging
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,SequentialSampler
import time
import datetime

train_data_path = "ratings_train2.txt"

dataset = pd.read_csv(train_data_path, sep="\t").dropna(axis=0)
text = list(dataset['document'].values)
label = dataset['label'].values

# 데이터 확인

num_to_print = 3
print("\n\n*** 데이터 ***")
for j in range(num_to_print):
    print(f'text : {text[j][:20]}, label: {label[j]}')
print(f'\t *학습 데이터의 수 : {len(text)}')
print(f'\t* 부정 리뷰 수 : {list(label).count(0)}')
print(f'\t* 긍정 리뷰 수 : {list(label).count(1)}')

# 토큰화

tokenizer = ElectraTokenizer.from_pretrained("koelectra-small-v3-discriminator")
inputs = tokenizer(text, truncation=True, max_length=256,add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n\n*** 토큰화 ***")
for j in range(num_to_print):
    print(f'\n{j+1}번째 데이터')
    print("** 토큰 **")
    print(input_ids[j])
    print("** 어텐션 마스크 **")
    print(attention_mask[j])

train, validation, train_y, validation_y = train_test_split(input_ids, label, test_size=0.2, random_state=2023)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, label, test_size=0.2, random_state=2023)

print("\n\n*** 데이터 분리 ***")
print(f"학습 데이터 수 : {len(train)}")
print(f"검증 데이터 수 : {len(validation)}")

batch_size = 16
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks  = torch.tensor(train_masks)
train_data   = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks =  torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = ElectraForSequenceClassification.from_pretrained('koelectra-small-v3-discriminator', num_labels=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-06)

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader)*epoch)

for e in range(0, epoch):
    print(f'\n\nEpoch {e+1} of {epoch}')
    print(f'** 학습 ** ')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed_rounded = int(round(time.time() - t0))
            elapsed = str(datetime.timedelta(seconds=elapsed_rounded))
            print(f'Batch {step} of {len(train_dataloader)}, 걸린 시간 : {elapsed}')

        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        model.zero_grad()
        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        if step % 10 == 0 and not step == 0:
            print(f'step : {step}, loss : {loss.item()}')
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"평균 학습 오차(loss) : {avg_train_loss}")
        epoch_elapsed_time = int(round(time.time() - t0))
        print(f"학습에 걸린 시간 : {str(datetime.timedelta(seconds=epoch_elapsed_time))}")
