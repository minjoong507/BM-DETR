# Usage: bash features/text_feature/text_feature.sh charades
dset_name=$1
gt_path=data
save_path=features

PYTHONPATH=$PYTHONPATH:. python utils/text_feature/extract.py \
--dset_name ${dset_name} \
--gt_path ${gt_path} \
--save_path ${save_path} \
${@:2}