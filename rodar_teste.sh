python .\main.py \
--scale_factor=2\
--mode=train \
--num_workers=0 \
--total_step=50000 \
--model_save_step=1000 \
--sample_step=100 \
--num_channels=6 \
--batch_size=1 \
--image_size=64 \
--trained_model=None

#Para teste no test set (test_and_error)

python .\main.py --scale_factor=2 --mode='test' --use_test_set='yes' --num_workers=0 --total_step=50000 --model_save_step=1000 --sample_step=100 --num_channels=6 --batch_size=1 --image_size=64 --trained_model=50000.pth.tar

#Para teste for do test set

python .\main.py --scale_factor=2 --mode='test' --use_test_set='no' --num_workers=0 --total_step=60000 --model_save_step=1000 --sample_step=100 --num_channels=6 --batch_size=1 --image_size=64 --trained_model=50000.pth.tar --test_image_path='./test_images/___lr___.png'