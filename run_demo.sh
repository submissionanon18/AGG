
#------------for text-guided semantic image translation -------------------
# CUDA_VISIBLE_DEVICES=0 python main_forgit.py -p "Desert" -s "Mountain" -i "./input_example/mountain.jpg" --output_path "./outputs/output_mo2de" --iterations_num 5 --use_prog_contrast --regularize_content --ddim--use_reverse_ddim --use_dds 


#------------for style transfer -------------------
# CUDA_VISIBLE_DEVICES=0 python main_forgit.py -p "A van gogh style painting" -s "A Photo" -i "input_example/day.jpg" --output_path "./outputs/output_photo2gogh" --use_reverse_ddim --iterations_num 5 --use_prog_contrast --regularize_content --gt_lambda 10 --ddim


#---------------for editing -----------------------
# CUDA_VISIBLE_DEVICES=0 python main_forgit.py -p "A lion is happy" -s "A lion" -i "input_example/lion1.jpg" --output_path "./outputs/output_lionhappy" --use_reverse_ddim --iterations_num 5 --use_prog_contrast --regularize_content --ddim --eta_edit 0 --eta 0 --eta_boost 0

#----------------for ffhq---------------------------
# CUDA_VISIBLE_DEVICES=0 python main_forgit.py -p "An Angry face" -s "A face" -i "input_example/face.jpeg" --output_path "./outputs/output_angry" --use_reverse_ddim --iterations_num 5 --clip_guidance_lambda 200 --use_prog_contrast --skip_timesteps 40 --lpips_lambda 0 --regularize_content --gt_lambda 200 --vit_lambda 0 --ddim --eta 0.0 --vit_lambda 0 --use_ffhq --id_lambda 0 --eta_edit 0.0 --timestep_respacing 100 --t_edit 20

#------------for image-guided image translation-------------------
# CUDA_VISIBLE_DEVICES=0 python main_forgit.py -p " " -s " " -i "input_example/dog_ref1.jpg" --output_path "./outputs/output_dog2tiger_onesot" -t "input_example/cat3.jpg" --use_reverse_ddim --iterations_num 5 --clip_guidance_lambda 0 --use_prog_contrast --skip_timesteps 40 --regularize_content --gt_lambda 200 --vit_lambda 1 --ddim --timestep_respacing 100 --lambda_trg 200 --l2_trg_lambda 200 --eta_edit 0.7 --use_noise_aug_all