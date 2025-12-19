export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.5 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Swan  --data_path dataset/NIPS_TS_Swan --input_c 38    --output_c 38 --patch2_size 25
python main.py --anomaly_ratio 0.5  --num_epochs 3      --batch_size 256     --mode test    --dataset NIPS_TS_Swan   --data_path dataset/NIPS_TS_Swan  --input_c 38    --output_c 38  --pretrained_model 20 --patch2_size 25