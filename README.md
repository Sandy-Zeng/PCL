# Proposal Contrastive Learning

Code for  the paper "Towards Effective Image Manipulation Detection with Proposal Contrastive Learning"

# Environment

tensorflow 0.12.1, python3.5.2, cuda 8.0.44 cudnn 5.1

As the codebase is adopted from RGB-N (Learning Rich Features for Image Manipulation Detection) the requirements of the environment and the sythetic dataset can be found from https://github.com/pengzhou1108/RGB-N). 

# Pre-trained model

For ImageNet resnet101 pre-trained model, please download from https://github.com/endernewton/tf-faster-rcnn

# Train on synthetic dataset

1. Change the data path in `lib/factory.py`:

   ```python
   coco_single_path= #FIXME
   for split in ['coco_train_filter_single', 'coco_test_filter_single']:
       name = split
       __sets[name] = (lambda split=split: coco(split,2007,coco_path))
   ```

2. Specify the ImageNet resnet101 pretrain model path in `train_faster_rcnn.sh` as below:

   ```python
   WEIGHT_PATH= #FIXME
   ```

3. Specify the dataset, gpu, and network in `train_dist_faster.sh` as below as run the file

   **(1) RGB-N**

   ```python
   ./train_faster_rcnn.sh 0 coco res101_fusion 0 EXP_DIR coco_res101_fusion
   ```

   **(2) RGB-N+PCL** 

   ```python
   ./train_faster_rcnn.sh 0 coco res101_contrastive 0 EXP_DIR coco_res101_contrastive
   ```

   **(3) RGB-C** 

   ```python
   ./train_faster_rcnn.sh 0 coco res101_constrained 0 EXP_DIR coco_res101_constrained
   ```

   **(4) RGB-C+PCL**

   ```python
   ./train_faster_rcnn.sh 0 coco res101_constrained_ssl 0 EXP_DIR coco_res101_constrained_ssl
   ```

# Use synthetic pre-trained model for fine tuning

 Take NIST dataset for examples  

## Supervised learning Setting

1. Change the data path in `lib/factory.py`:

   ```python
   nist_path= #FIXME
   for split in ['dist_NIST_train_new_2','dist_NIST_test_new_2']:
       name = split
       __sets[name] = (lambda split=split: nist(split,2007,nist_path))
   ```

2. Specify the RGB-C+PCL coco pretrain model path in `train_faster_rcnn.sh` as below:

   ```python
   WEIGHT_PATH= #FIXME
   ```

3. Specify the dataset, gpu, and network in `train_dist_faster.sh` as below as run the file:

   **RGB-C+PCL** 

   ```
   ./train_faster_rcnn.sh 0 NIST res101_constrained_ssl 0 EXP_DIR nist_res101_constrained_ssl
   ```

## Semi-Supervised learning Setting

1. Set the label of unlabled data as **"semi"** and label of labeled data as **"tamper"**

   example data shown in ./data/NIST_train_new_2_semi.txt

2. Change the data path in `lib/factory.py`:

   ```python
   nist_path= #FIXME
   for split in ['dist_NIST_train_new_2_semi', 'dist_NIST_test_new_2']:
       name = split
       __sets[name] = (lambda split=split: nist(split,2007,nist_path))
   ```

3. Specify the RGB-C+PCL coco pretrain model path in `train_faster_rcnn.sh` as below:

   ```python
   WEIGHT_PATH= #FIXME
   ```

4. Specify the dataset, gpu, and network as below and run the command:

   **RGB-C+PCL** 

   ```python
   ./train_faster_rcnn.sh 0 NIST_unlabeled res101_constrained_ssl 0 EXP_DIR nist_res101_constrained_ssl_semi
   ```

# Test the model

1. Check the model path match well, making sure the checkpoint iteration exist in model output path. 

   ```python
     coco)
       TRAIN_IMDB="coco_train_filter_single"
       TEST_IMDB="coco_test_filter_single"
       ITERS=110000
       ANCHORS="[8,16,32,64]"
       RATIOS="[0.5,1,2]"
       ;;
   ```

2. Specify the dataset, gpu, network and iters as below and run the command:

   ```python
   ./test_faster_rcnn.sh 0 coco res101_constrained_ssl 110000 EXP_DIR coco_res101_constrained_ssl
   ```

# Citation:

