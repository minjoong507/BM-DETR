dset_name=hl
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results/results_${dset_name}/${v_feat_types}
feat_root=features
ctx_mode=video_tef
exp_id=exp

######## data paths
train_path=data/QVHighlights/train.jsonl
eval_path=data/QVHighlights/val.jsonl
eval_split_name=val
stop_metric=MR-full-mAP

######## setup video+text features
# video features
v_feat_dim=0
v_feat_dirs=()

if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/hl_features/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/hl_features/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
t_feat_dir=${feat_root}/hl_features/clip_text_features/
t_feat_dim=512

echo "QVHighlights, feat types: " ${v_feat_types} ${t_feat_type}

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
--n_epoch 200 \
--lr 0.0001 \
--lr_drop 400 \
--lw_saliency 1 \
--clip_length 2 \
--max_v_l 75 \
--max_q_l 32 \
--bsz 32 \
--lw_prob_loss_coef 2 \
--lw_contrastive_loss_coef 3 \
${@:1}
