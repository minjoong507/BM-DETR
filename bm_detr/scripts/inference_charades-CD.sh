ckpt_path=$1
eval_split_name=$2

if [[ ${eval_split_name} == "test_iid" ]]; then
  eval_path=data/charades-CD/test_iid.jsonl
elif [[ ${eval_split_name} == "test_ood" ]]; then
  eval_path=data/charades-CD/test_ood.jsonl
else
    echo "Wrong Test set."
    exit 1
fi

PYTHONPATH=$PYTHONPATH:. python bm_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
