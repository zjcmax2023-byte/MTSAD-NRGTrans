export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.9 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25 --patch2_size 50
python main.py --anomaly_ratio 0.9  --num_epochs 3        --batch_size 256     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20 --patch2_size 50

