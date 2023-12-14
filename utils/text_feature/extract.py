import os
from utils.clip import clip
from utils.basic_utils import load_jsonl, dict_to_markdown
import torch
import numpy as np
from tqdm import tqdm
import json

def extract_clip_text_features(args, phase):
    model, _ = clip.load(args.model, device=args.device, jit=False)
    model.eval()
    gt = load_jsonl(args.gt_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for data in tqdm(gt, total=len(gt), desc='Extract text features {} {}'.format(args.dset_name, phase)):
        file_path = os.path.join(args.save_path, '{}{}'.format(args.format, data['qid']))
        if os.path.exists(f"{file_path}.npz"):
            raise FileExistsError('{} already exists!'.format(file_path))
        np.savez(os.path.join(args.save_path, '{}{}'.format(args.format, data['qid'])), **_get_clip_text_features(model, data['query'], args.device))
    print('done.')

def _get_clip_text_features(model, text, device):
    clip_texts = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(clip_texts)
        for k, v in text_features.items():
            if k == 'last_hidden_state':
                v = v.squeeze(0)
            text_features[k] = v.cpu().numpy()
    return text_features
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text Features Extraction Script")
    parser.add_argument("--dset_name", type=str, default='charades',
                        help="target dataset name")
    parser.add_argument("--gt_path", type=str, default=None, help="path to GT json file")
    parser.add_argument("--phase", type=list, default=['train', 'val', 'test'], help="path to GT json file")
    parser.add_argument("--save_path", type=str, default='None', help="path to save the results")
    parser.add_argument("--device", type=str, default=None, help="path to save the results")
    parser.add_argument("--format", type=str, default='qid', help="format of filename")
    parser.add_argument('--model', type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-L/32"])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # Download QVHighlights features from https://github.com/jayleicn/moment_detr.
    if args.dset_name not in ['charades', 'charades-CD', 'activitynet', 'tacos']:
        raise NotImplementedError('Invalid dataset, {} '.format(args.dset_name))

    # There is no val set in charades-sta dataset.
    args.phase = ['train', 'test']
    if 'activitynet' in args.dset_name:
        args.phase = ['train', 'val_1', 'val_2']
    elif 'tacos' in args.dset_name:
        args.phase = ['train', 'val', 'test']

    if 'CD' in args.dset_name:
        args.phase = ['train', 'val', 'test_iid', 'test_ood']

    print(dict_to_markdown(vars(args), max_str_len=120))

    args.save_path = os.path.join(args.save_path, f'{args.dset_name}_features', 'clip_text_features')
    for phase in args.phase:
        args.gt_path = os.path.join('data', args.dset_name, f'{phase}.jsonl')
        extract_clip_text_features(args, phase)


if __name__ == '__main__':
    main()
