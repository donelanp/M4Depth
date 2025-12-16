SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR=${SCRIPT_DIR}/..
CKPT_DIR=${BASE_DIR}/ckpt/usegeo

values=(1 2 3 4 5 6 7 8 9 10 20 30 40 50)
for i in "${values[@]}"; do
    idx=$(printf "%04d" $i)

    echo "model_checkpoint_path: \"cp-${idx}.ckpt\"" > ${CKPT_DIR}/train/checkpoint
    echo "all_model_checkpoint_paths: \"cp-${idx}.ckpt\"" >> ${CKPT_DIR}/train/checkpoint

    python main.py --mode=validation --dataset=usegeo "" --ckpt_dir=${CKPT_DIR} --records=${BASE_DIR}/data/usegeo/test_data --keep_top_n=100
done