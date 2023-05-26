## Official repository of "Improved Diffusion-based Image Translation using Asymmetric Gradient Guidance"

### Environment
Pytorch 1.9.0, Python 3.9

```
$ conda create --name AGG python=3.9
$ conda activate AGG
$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install color-matcher
$ pip install git+https://github.com/openai/CLIP.git
```

### Model download
To generate images, please download the pre-trained diffusion model

imagenet 256x256 [LINK](https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view?usp=sharing)

FFHQ 256x256 [LINK](https://drive.google.com/file/d/1-oY7JjRtET4QP3PIWg3ilxAo4VfjCa3J/view?usp=sharing)

download the model into ```./checkpoints``` folder

For face identity loss when using FFHQ pre-trained model, download pre-trained ArcFace model [LINK](https://drive.google.com/file/d/1SJa5qVNM6jGZdmsnUsGNhjtrssGYuJfT/view?usp=sharing)

save the model into ```./id_model```

### Inference 

Please refer to the bash script 
```
./run_demo.sh
```

For memory saving, we can use single CLIP model with ```--clip_models 'ViT-B/32'```

Our source code heavily rely on DiffuseIT
