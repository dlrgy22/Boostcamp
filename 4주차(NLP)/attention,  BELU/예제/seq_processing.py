from tqdm import tqdm

import torch
vocab_size = 100
pad_id = 0
sos_id = 1
eos_id = 2

def padding(data, is_src=True):
    max_len = len(max(data, key=len))

    valid_lens = []
    for i, seq in enumerate(tqdm(data)):
        valid_lens.append(len(seq))
        if len(seq) < max_len:
            data[i] = seq + [pad_id]*(max_len - len(seq))
    
    return torch.LongTensor(data), torch.LongTensor(valid_lens), torch.LongTensor(max_len)


if __name__ == "__main__":
    src_data = [
    [3, 77, 56, 26, 3, 55, 12, 36, 31],
    [58, 20, 65, 46, 26, 10, 76, 44],
    [58, 17, 8],
    [59],
    [29, 3, 52, 74, 73, 51, 39, 75, 19],
    [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
    [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
    [75, 34, 17, 3, 86, 88],
    [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
    [12, 40, 69, 39, 49]
    ]

    trg_data = [
    [75, 13, 22, 77, 89, 21, 13, 86, 95],
    [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
    [85, 8, 50, 30],
    [47, 30],
    [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
    [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
    [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
    [16, 98, 68, 57, 55, 46, 66, 85, 18],
    [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
    [37, 93, 98, 13, 45, 28, 89, 72, 70]
    ]

    trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)]
    src_data, src_lens, src_max_len = padding(src_data)
    trg_data, trg_lens, trg_max_len = padding(trg_data)
    print(type(trg_data))