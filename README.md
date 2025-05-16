# Memory Storyboard

Code for the paper ["Memory Storyboard: Leveraging Temporal Segmentation for Streaming Self-Supervised Learning from Egocentric Videos"](https://agenticlearning.ai/memory-storyboard/)(CoLLAs 2025).

## Installation
Run `pip install -r requirements.txt` to install requirements.

## Datasets

### Training Datasets

You can access the [SAYCam dataset](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00039/97495/SAYCam-A-Large-Longitudinal-Audiovisual-Dataset) through [Databrary](https://nyu.databrary.org/volume/564) after going through an authorization process. As of May 2025, the site is currently under maintainance but it should be back online soon. We are not allowed to redistribute the dataset or its derivatives here according to its license.

You can download the [KrishnaCam dataset](https://krsingh.cs.ucdavis.edu/krishna_files/papers/krishnacam/krishnacam.html) from its offical source [here](https://drive.google.com/drive/folders/1q81yrQenY1dMul3ixJUbOrf9FPOTgMGW). Then, use the script `metas/convert_video_frame.py` to decode the videos into frames at 10 fps.

### Evaluation Datasets
You can access the [Labeled-S dataset](https://arxiv.org/abs/2007.16189) through the same [Databrary repo](https://nyu.databrary.org/volume/564) as SAYCam above. Our training and test splits for Labeled-S can be found at `metas/labeledS_train_val.txt`, where each line includes the file name and its corresponding label. The first half of the file is the training set and the second half is the test set.

You can access the [ImageNet dataset (ILSVRC 2017)](https://image-net.org/download.php) through [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). Our training and test splits for mini-ImageNet can be found at `metas/miniinet_train_val.txt`, with similar syntax as the Labeled-S dataset above.

You can access the iNaturalist 2018 Dataset from their official [github repo](https://github.com/visipedia/inat_comp/blob/master/2018/README.md#Data).

## Training Memory Storyboard

Example command for training:
```
python training/train_storyboard_classmerging.py  \
        --save_dir  <path_to_save_checkpoints> \
        --curr_batch_size 64 \
        --replay_batch_size 448 \
        --long_buffer_size 45000 \
        --short_buffer_size 5000 \
        --curr_loss_coef 1.0 \
        --tc_loss_coef 1.0 \
        --group_norm \
        --depth 50 \
        --dataset saycam \
        --subsample 8 \
        --class_length 7500 \
        --lr 0.05 \
        --warmup 500 \
        --method simsiam \
        --merge_threshold 0.003
```

You can use the `--imagenet_eval` and `--labeledS_eval` flags to enable periodic SVM and kNN classification evaluation on the mini-ImageNet and Labeled-S datasets.

## Running Linear Probing Evaluations
You can run linear probing evaluation on the iNaturalist 2018 dataset with
```
python linear_decoding_inat.py --data <path_to_inat_data> --num_classes 8142 --epochs 20 --batch_size 1024 --fc_bn --image_size 112 --load_dir <path_to_model>
```

You can run linear probing evaluation on the ImageNet dataset with
```
python linear_decoding.py --data <path_to_imagenet> --num_classes 1000 --epochs 10 --batch_size 1024 --fc_bn --image_size 112 --load_dir <path_to_model> 
```

## Acknowledgements

We thank the authors of the papers ["How Well Do Unsupervised Learning Algorithms Model Human Real-time and Life-long Learning?"](https://github.com/neuroailab/VisualLearningBenchmarks), ["The Challenges of Continuous Self-Supervised Learning"](https://github.com/senthilps8/continuous_ssl_problem), and ["Integrating Present and Past in Unsupervised Continual Learning"](https://github.com/SkrighYZ/Osiris) for releasing their code.
