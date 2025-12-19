export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.8 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Water  --data_path dataset/NIPS_TS_Water  --input_c 9 --output_c 9 --patch2_size 50
python main.py --anomaly_ratio 0.8  --num_epochs 3     --batch_size 256   --mode test    --dataset NIPS_TS_Water   --data_path dataset/NIPS_TS_Water --input_c 9    --output_c 9 --pretrained_model 20 --patch2_size 50


