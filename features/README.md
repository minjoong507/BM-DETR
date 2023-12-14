# Features

1. Extract features files:

```
mkdir {dset_name}_features
tar -zxvf path/to/{feat_type}.tar.gz
```

2. Once the features are downloaded, you need to check that your folder structure is the same as below:

```angular2html
BM-DETR
└── features
    ├── charades_features
    │   ├── c3d_features
    │   ├── vgg_features
    │   ├── slowfast_features
    │   ├── clip_features
    │   └── clip_text_features
    ├── charades-CD_features
    │   ├── i3d_features
    │   └── clip_text_features
    ├── hl_features
    │   ├── slowfast_features
    │   ├── clip_features
    │   └── clip_text_features
    ├── tacos_features
    │   ├── c3d_features
    │   └── clip_text_features
    └── activitynet_features
        ├── c3d_features
        └── clip_text_features
```
