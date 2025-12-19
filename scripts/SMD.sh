export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.4 --num_epochs 6   --batch_size 256  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38 --patch2_size 5
python main.py --anomaly_ratio 0.4 --num_epochs 6   --batch_size 256     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20 --patch2_size 5