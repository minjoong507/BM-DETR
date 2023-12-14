dset_name=charades
v_feat_types=$1
t_feat_type=clip
results_root=results/results_${dset_name}/${v_feat_types}
feat_root=features
ctx_mode=video_tef
exp_id=exp

######## data paths
train_path=data/charades/train.jsonl
eval_path=data/charades/test.jsonl
eval_split_name=test
stop_metric=MR-full-R1@0.7

######## setup video+text features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
    v_feat_dirs+=(${feat_root}/charades_features/slowfast_features)
    (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
    v_feat_dirs+=(${feat_root}/charades_features/clip_features)
    (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"c3d"* ]]; then
    v_feat_dirs=(${feat_root}/charades_features/c3d_features)
    (( v_feat_dim = 1024 ))
fi
if [[ ${v_feat_types} == *"vgg"* ]]; then
    v_feat_dirs=(${feat_root}/charades_features/vgg_features)
    (( v_feat_dim = 4096 ))
fi

# text features
t_feat_dir=${feat_root}/charades_features/clip_text_features/
t_feat_dim=512

echo "Charades-STA, feat types: " ${v_feat_types} ${t_feat_type}

PYTHONPATH=$PYTHONPATH:. python bm_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--v_feat_type ${v_feat_types} \
--t_feat_type ${t_feat_type} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--stop_metric ${stop_metric} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--clip_length 1 \
--bsz 32 \
--max_v_l 75 \
--max_q_l 32 \
--n_epoch 100 \
--lw_saliency 4 \
--lr 0.0002 \
--lr_drop 40 \
--use_temporal_shifting \
${@:2}
