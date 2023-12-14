# Preparation

## Feature Extraction

1. Video features

We follow [Linjie's HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor) to extract video features.
If you want to use your own features, please download the raw videos of each dataset.

2. Text features

All of annotation files should be located in `data`.

```angular2html
BM-DETR
└── data
    ├── charades
    │   └── train.jsonl
    │   └── test.jsonl
    ├── charades-CD
    │   └── train.jsonl
    │   └── val.jsonl
    │   └── test_iid.jsonl
    │   └── test_ood.jsonl
    ├── QVHighlights
    │   └── train.jsonl
    │   └── val.jsonl
    │   └── test_public.jsonl
    ├── tacos
    │   └── train.jsonl
    │   └── val.jsonl
    │   └── test.jsonl
    └── activitynet
        └── train.jsonl
        └── val_1.jsonl
        └── val_2.jsonl
```

You can extract the text features for each dataset:
```angular2html
bash utils/extract_text_features/extract_text_features.sh {dset_name}
```

It can be one of `charades`, `charades-CD`, `tacos`, and `activitynet`.

For `QVHighlights`, please download official features from [here](https://github.com/jayleicn/moment_detr).

Please see the `extract_text_features.sh` to check the save path.