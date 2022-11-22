import torch
from torch import tensor
from warcraft_shortest_path.metrics import is_valid_label_fn_new

def parse_results(fname):
    f = open(fname)
    lines = f.readlines()[:-1]
    mode = -1 # 0 = true, 1 = pred
    tensors = [None,None]
    strings = ['','']
    for l in lines:
        if 'grad_fn=<RoundBackward>)' in l or 'True' in l or 'test' in l or 'loss' in l:
            continue
        if l[:6] == 'tensor':
            if mode != -1 and strings[mode] != '':
                cur_tensor = eval(strings[mode][:-19]+')')
                tensors[mode] = cur_tensor if tensors[mode] is None else torch.cat((tensors[mode],cur_tensor))
                strings[mode] = ''
            mode = 1-mode if mode != -1 else 0
        if mode != -1:
            strings[mode] += l
    cur_tensor = eval(strings[mode][:-19]+')')
    tensors[mode] = cur_tensor if tensors[mode] is None else torch.cat((tensors[mode],cur_tensor))
    return (tensors[0],tensors[1])

def percent_valid_paths(fname):
    true_paths, pred_paths = parse_results(fname)
    pred_count = 0
    for i in range(pred_paths.shape[0]):
        if is_valid_label_fn_new(pred_paths[i]):
            pred_count += 1
    return pred_count/pred_paths.shape[0]