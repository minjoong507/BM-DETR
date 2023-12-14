ckpt_path=$1

eval_path1=data/QVHighlights/val.jsonl
eval_path2=data/QVHighlights/test_public.jsonl

PYTHONPATH=$PYTHONPATH:. python bm_detr/inference.py \
--resume ${ckpt_path}/model_best.ckpt \
--eval_split_name val \
--eval_path ${eval_path1} \
${@:3}; sleep 3

PYTHONPATH=$PYTHONPATH:. python bm_detr/inference.py \
--resume ${ckpt_path}/model_best.ckpt \
--eval_split_name test_public \
--eval_path ${eval_path2} \
${@:3}; sleep 3

echo "Move prediction files to submission folder."
cp ${ckpt_path}/inference_hl_test_public_preds.jsonl submission/hl_test_submission.jsonl
cp ${ckpt_path}/inference_hl_val_preds.jsonl submission/hl_val_submission.jsonl
echo "done."
