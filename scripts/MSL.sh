export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.9 --num_epochs 6   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55 --patch2_size 20
python main.py --anomaly_ratio 0.9  --num_epochs 6      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20 --patch2_size 20
