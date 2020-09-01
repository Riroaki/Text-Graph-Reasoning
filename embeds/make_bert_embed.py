import numpy as np
import torch
import sys
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

cuda_device = torch.device('cuda:0')
cpu_device = torch.device('cpu')

pretrain_model_path = './raw-bert'
if len(sys.argv) >= 2:
    pretrain_model_path = sys.argv[1]
embed_path = 'emb_raw.npz'
if len(sys.argv) >= 3:
    embed_path = sys.argv[2]
max_seq_length = 150

tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
model = BertModel.from_pretrained(pretrain_model_path).to(cuda_device)


def process(batch_text):
    batch_ids, batch_masks, batch_seg_ids = [], [], []
    for single_text in batch_text:
        tokens = ["[CLS]"] + \
            tokenizer.tokenize(single_text)[:max_seq_length - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        batch_ids.append(input_ids)
        batch_masks.append(input_mask)
        batch_seg_ids.append(segment_ids)
    batch_ids = torch.tensor(batch_ids, dtype=torch.long).to(cuda_device)
    batch_masks = torch.tensor(batch_masks, dtype=torch.long).to(cuda_device)
    batch_seg_ids = torch.tensor(batch_seg_ids,
                                 dtype=torch.long).to(cuda_device)

    with torch.no_grad():
        emb = model(input_ids=batch_ids,
                    token_type_ids=batch_seg_ids,
                    attention_mask=batch_masks)[0][:, 0]

    return emb.cpu().numpy()


embed = []
with open('../data/reduce_all/entity2text.txt', 'r') as f:
    batch, batch_size = [], 600
    for line in tqdm(f.readlines(), desc='encoding texts'):
        entity, text = line.strip().split('\t')
        batch.append(text)
        if len(batch) == batch_size:
            embed.append(process(batch))
            batch = []
if len(batch) > 0:
    embed.append(process(batch))
embed = np.concatenate(embed)
print(embed.shape)
np.savez_compressed(embed_path, embed=embed)
