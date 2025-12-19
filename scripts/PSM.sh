export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.9 --num_epochs 6    --batch_size 256  --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25 --patch2_size 25
python main.py --anomaly_ratio 0.9  --num_epochs 6       --batch_size 256     --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20 --patch2_size 25

