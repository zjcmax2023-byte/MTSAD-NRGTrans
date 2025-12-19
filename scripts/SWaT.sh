export CUDA_VISIBLE_DEVICES=0

python main.py --anomaly_ratio 0.9 --num_epochs 6    --batch_size 256  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51 --patch2_size 25
python main.py --anomaly_ratio 0.9  --num_epochs 6    --batch_size 256     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --pretrained_model 10 --patch2_size 25
