# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob
import pdb
from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from sklearn import metrics
# from sklearn.metrics import roc_auc_score
from PIL import Image
import matplotlib.pyplot as plt


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  row,col,ch= im.shape
  mean = 0
  var = 10
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)


  # jpeg
  #cv2.imwrite('b.jpg',im,[cv2.IMWRITE_JPEG_QUALITY, 70])
  #pdb.set_trace()
  #im=cv2.imread('b.jpg')

  im_orig = im.astype(np.float32, copy=True)
  #im_orig = im_orig + gauss
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  processed_noise = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    noise=im
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
    processed_noise.append(noise)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  noise_blob = im_list_to_blob(processed_noise)
  return blob,noise_blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'],blobs['noise'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.

  if cfg.USE_MASK is True:
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0], im.shape[0], im.shape[1]]],
                                    dtype=np.float32)
    scores1, scores, bbox_pred, rois, feat, s, y_preds, mask_data,layers = net.test_image(sess, blobs['data'], blobs['noise'],
                                                                                       blobs['im_info'])
        
    boxes = rois[:, 1:5] / im_scales[0]
    mask_boxes = mask_data[:, 1:5]/im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
    mask_scores = np.reshape(mask_data[:, 5], [mask_data[:, 5].shape[0], -1])
    mask_boxes = np.reshape(mask_boxes, [mask_boxes.shape[0], -1])
    maskcls_ind = np.reshape(mask_data[:, -1], [mask_data[:, -1].shape[0], -1])

    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred
      pred_boxes = bbox_transform_inv(boxes, box_deltas)
      pred_boxes = _clip_boxes(pred_boxes, im.shape)
      pred_mask_boxes = mask_boxes
      # pred_boxes = _clip_boxes(boxes, im.shape)
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))
      pred_mask_boxes = np.tile(mask_boxes, (1, mask_scores.shape[1]))
    return scores, pred_boxes, feat, s, maskcls_ind, pred_mask_boxes, mask_scores, y_preds, mask_data, layers
  else:
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    try:
      scores1, scores, bbox_pred, rois,feat,s = net.test_image(sess, blobs['data'], blobs['im_info'])
    except:
      scores1, scores, bbox_pred, rois,feat,s = net.test_image(sess, blobs['data'],blobs['noise'], blobs['im_info'])
    boxes = rois[:, 1:5] / im_scales[0]
    # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred
      pred_boxes = bbox_transform_inv(boxes, box_deltas)
      pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes, feat, s

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


def cal_precision_recall_mae(prediction, gt):
  # input should be np array with data type uint8
  # assert prediction.dtype == np.uint8
  # assert gt.dtype == np.uint8
  assert prediction.shape == gt.shape
  y_test = gt.flatten()
  y_pred = prediction.flatten()
  precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
  # auc_score = roc_auc_score(y_test, y_pred, average='micro')
  fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, pos_label=1)
  # print(thresholds)
  # print(y_test)
  # print(y_pred)
  # assert False
  auc_score = metrics.auc(fpr, tpr)
  return precision, recall, auc_score, thresholds, tpr, fpr


def cal_fmeasure(precision, recall):
  # max_fmeasure = max([(2 * p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)])
  max_fmeasure = [(2 * p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
  return max_fmeasure


def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  all_f1 = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  all_auc = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  all_auc_new = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  counters = []

  output_dir = get_output_dir(imdb, weights_filename)
  if os.path.isfile(os.path.join(output_dir, 'detections.pkl')):
    all_boxes = pickle.load(open(os.path.join(output_dir, 'detections.pkl'), 'rb'))
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    return
    print(np.array(all_boxes).shape)
    for i in range(np.array(all_boxes).shape[1]):
      test_boxes = [[[] for _ in range(1)] for _ in range(2)]
      test_boxes[1][0] = all_boxes[1][i]
      print(test_boxes)
      print(np.array(test_boxes).shape)
      imdb.evaluate_detections(test_boxes, output_dir)
    imdb.evaluate_detections(all_boxes, output_dir)
    return

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    if cfg.USE_MASK == False:
      scores, boxes,_ ,_ = im_detect(sess, net, im)
    else:
      scores, boxes, feat, s, maskcls_inds, mask_boxes, mask_scores, mask_pred, _ ,_= im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                      for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time,
              _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
  
    

def test_auc_f1(imdb, weights_filename, threshold):
  output_dir = get_output_dir(imdb, weights_filename)
  all_boxes = pickle.load(open(os.path.join(output_dir, 'detections.pkl'), 'rb'))
  all_f1 = np.zeros(imdb.num_images, np.float)
  all_auc = np.zeros(imdb.num_images, np.float)
  num_images = len(imdb.image_index)
  error_num = 0
  output_dir = get_output_dir(imdb, weights_filename)
  result_file = os.path.join(output_dir, 'auc_log.txt')
  f = open(result_file, 'w')
  for i in range(num_images):
    # pred mask
    im_path = imdb.image_path_at(i)
    im = cv2.imread(im_path)
    h, w = im.shape[0], im.shape[1]
    pred_mask = np.zeros((h, w))
    # print(len(all_boxes[1][i]))
    pred_boxes = all_boxes[1][i]
    pred_boxes = sorted(pred_boxes, key=lambda j: j[4])[::-1]
    # pred_boxes = pred_boxes[1:2]
    # print(len(pred_boxes))
    evalate_boxes = [[[]] for _ in range(imdb.num_classes)]
    evalate_boxes[1][0] = all_boxes[1][i]
    for idx, j in enumerate(pred_boxes):
      # print(j[4])
      if j[4] < threshold:
        continue
      # if idx >= 1:
      #   break
      x1,y1,x2,y2 = int(j[0]),int(j[1]),int(j[2]),int(j[3])
      temp = np.zeros((h, w))
      # print(x1, y1, x2, y2)
      # print(j[4])
      temp[y1:y2+1, x1:x2+1] = j[4]
      pred_mask = np.where(pred_mask >= temp, pred_mask, temp)
    # gt mask
    mask_path = imdb.mask_path_at(i)
    if mask_path == 0:
      error_num += 1
      continue
    gt_mask = cv2.imread(mask_path)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    ret, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    gt_mask = (gt_mask / 255.0).astype(np.float32)
    if im_path == '/data/zengyuyuan/Project/RGB-N/CASIA/CASIA1/Sp/Sp_D_CNN_A_sec0012_ani0007_0275.jpg':
      gt_mask = np.transpose(gt_mask)
    # f1 auc
    try:
      prec, recall, auc_score = cal_precision_recall_mae(pred_mask, gt_mask)
      f1 = cal_fmeasure(prec, recall)
      all_f1[i] = f1
      all_auc[i] = auc_score
      # print("image path: {}, f1: {}, auc_score:{} ".format(os.path.basename(im_path), f1, auc_score))
      # print(im_path)
      # print(f1, auc_score)
      # ap, map = imdb.evaluate_detections(evalate_boxes, output_dir)
      # print("image path: {}, f1: {}, auc_score:{} ap:{}, map:{} ".format(os.path.basename(im_path), f1, auc_score, ap, map))
      # f.write("image path: {}, f1: {}, auc_score:{} ap:{}, map:{} \n".format(os.path.basename(im_path), f1, auc_score, ap, map))
      # assert False
    except Exception as e:
      print('error occurs in {}'.format(im_path))
      print(e)
      print(pred_mask.shape)
      print(gt_mask.shape)
    test_images = [imdb.image_index[i]]
    ap, map = imdb.evaluate_detections(evalate_boxes, test_images, output_dir)
    print("image path: {}, f1: {}, auc_score:{} ap:{}, map:{} ".format(os.path.basename(im_path), f1, auc_score, ap, map))
    f.write(
      "image path: {}, f1: {}, auc_score:{} ap:{}, map:{} \n".format(os.path.basename(im_path), f1, auc_score, ap, map))
    # if i > 4:
    #   assert False
  f1_ind = np.where(all_f1 > 0.)[0]
  all_f1 = all_f1[f1_ind]
  auc_ind = np.where(all_f1 > 0.)[0]
  all_auc = all_auc[auc_ind]
  f.close()

  print("{} mask file does not exist".format(error_num))
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('Test Results:')
  print("keep box theshold: {}".format(threshold))
  print("f1 valid: {}".format(len(all_f1)))
  print("auc valid: {}".format(len(all_auc)))
  print('Average F1  Score: %.3f' % np.average(all_f1))
  print('Average AUC Score: %.3f' % np.average(all_auc))
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
  
  
def test_micro_auc_f1(imdb, weights_filename, threshold):
  output_dir = get_output_dir(imdb, weights_filename)
  all_boxes = pickle.load(open(os.path.join(output_dir, 'detections.pkl'), 'rb'))
  all_masks = []
  all_prediced_masks = []
  num_images = len(imdb.image_index)
  error_num = 0
  for i in range(num_images):
    # pred mask
    im_path = imdb.image_path_at(i)
    im = cv2.imread(im_path)
    h, w = im.shape[0], im.shape[1]
    pred_mask = np.zeros((h, w))
    # print(len(all_boxes[1][i]))
    for idx, j in enumerate(all_boxes[1][i]):
      if j[4] < threshold:
        continue
      # if idx > 1:
      #   break
      x1,y1,x2,y2 = int(j[0]),int(j[1]),int(j[2]),int(j[3])
      temp = np.zeros((h, w))
      # print(x1, y1, x2, y2)
      # print(j[4])
      temp[y1:y2+1, x1:x2+1] = j[4]
      pred_mask = np.where(pred_mask >= temp, pred_mask, temp)
    # gt mask
    mask_path = imdb.mask_path_at(i)
    if mask_path == 0:
      # print('error')
      # print(mask_path)
      error_num += 1
      continue
    gt_mask = cv2.imread(mask_path)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    ret, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    gt_mask = (gt_mask / 255.0).astype(np.float32)
    
    all_masks.extend(np.reshape(gt_mask, -1))
    all_prediced_masks.extend(np.reshape(pred_mask, -1))
  
  all_masks = np.array(all_masks)
  all_prediced_masks = np.array(all_prediced_masks)
  # print(all_masks.shape)
  # print(all_prediced_masks.shape)
  # f1 auc
  try:
    prec, recall, auc_score = cal_precision_recall_mae(all_prediced_masks, all_masks)
    f1 = cal_fmeasure(prec, recall)
  except Exception as e:
    print('error occurs')
    print(e)

  print("{} mask file does not exist".format(error_num))
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('Test Results:')
  print("keep box theshold: {}".format(threshold))
  print('Average F1  Score: %.3f' % f1)
  print('Average AUC Score: %.3f' % auc_score)
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')


def test_mask(sess, net, imdb, weights_filename, max_per_image=100, thresh=0):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)] for _ in range(imdb.num_classes)]
  all_f1 = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  all_auc = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  all_auc_new = np.zeros((imdb.num_images, imdb.num_classes), np.float)
  counters = []

  output_dir = get_output_dir(imdb, weights_filename)
 
  _t = {'im_detect': Timer(), 'mask': Timer()}
  for i in range(num_images):
    print(imdb.image_path_at(i))
    im_path = imdb.image_path_at(i)
    im = cv2.imread(im_path)

    mask_gt = cv2.imread(imdb.mask_path_at(i))
    mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
    ret, mask_gt = cv2.threshold(mask_gt, 127, 255, cv2.THRESH_BINARY)
    mask_gt = (mask_gt / 255.0).astype(np.float32)

    _t['im_detect'].tic()
    scores, boxes, feat, s, maskcls_inds, mask_boxes, mask_scores, mask_pred, _ ,_= im_detect(sess, net, im)
    _t['im_detect'].toc()
    _t['mask'].tic()

     # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                      for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]

    batch_ind = np.where(mask_scores > 0.)[0]
    cls=maskcls_inds[np.argmax(mask_scores)].astype(int)
    mask_boxes=mask_boxes.astype(int)
    if batch_ind.shape[0] == 0:
      f1 = 1e-10
      auc_score = 1e-10
    else:
      mask_out = np.zeros(im.shape[:2],dtype=np.float)
      for ind in batch_ind:
        height = mask_boxes[ind, 3] - mask_boxes[ind, 1]
        width = mask_boxes[ind, 2] - mask_boxes[ind, 0]
        if width <= 0 or height <= 0:
          continue
        else:
          mask_box_pre = cv2.resize(mask_pred[ind, :, :, :], (width, height))
          mask_pre = np.zeros(im.shape[:2],dtype=np.float)
          bbox1 = mask_boxes[ind, :]
          mask_pre[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = mask_box_pre
          mask_out = np.where(mask_out >= mask_pre, mask_out, mask_pre)
      if im_path == '/data/zengyuyuan/Project/RGB-N/CASIA/CASIA1/Sp/Sp_D_CNN_A_sec0012_ani0007_0275.jpg':
        mask_gt = np.transpose(mask_gt)
      if mask_out.shape != mask_gt.shape:
        print(mask_out.shape)
        print(mask_gt.shape)
        # mask_gt = np.resize(mask_gt, mask_out.shape)
        continue 
      precision, recall, auc_score, thresholds, tpr, fpr = cal_precision_recall_mae(mask_out, mask_gt)
      f1 = cal_fmeasure(precision, recall)
      # print(f1)
      f1_inds = np.argmax(np.array(f1))
      f1=np.max(np.array(f1))
      thres = thresholds[f1_inds]
      mask_pred = (mask_out > thres).astype(np.float32)*255.0

      img_name = os.path.basename(im_path)
      mask_name = os.path.basename(im_path).split('.')[0] + '_%.2f_%.2f'%(f1, auc_score)+'.jpg'
      # cv2.imwrite(os.path.join('/data/zengyuyuan/Project/RGBN_SSL/nist_gt/', img_name), (mask_gt*255.0).astype(np.uint8))
      # cv2.imwrite(os.path.join('/data/zengyuyuan/Project/RGBN_SSL/nist_nossl_predict/', mask_name), mask_pred.astype(np.uint8))
      # plt.figure()
      # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc_score), lw=2)
      # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
      # plt.ylim([-0.05, 1.05])
      # plt.xlabel('False Positive Rate')
      # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
      # plt.title('ROC Curve')
      # plt.legend(loc="lower right")
      # plt.savefig(os.path.join('/mnt/wx_feature/home/yuyuanzeng/PS/RGBN_SSL/predict_mask/imdb_roc_curve_nossl', mask_name))

      plt.figure()
      plt.plot(recall, precision, 'k--', label='F1 {0:.2f}'.format(f1), lw=2)
      plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
      plt.ylim([-0.05, 1.05])
      plt.xlabel('Recall')
      plt.ylabel('Precision')  # 可以使用中文，但需要导入一些库即字体
      plt.title('PR Curve')
      plt.legend(loc="lower right")
      plt.savefig(os.path.join('/mnt/wx_feature/home/yuyuanzeng/PS/RGBN_SSL/predict_mask/imdb_pr_curve', mask_name))

    print('F1 score per image:', f1)
    print('AUV score per image:', auc_score)
    all_f1[i, cls] = f1
    all_auc[i, cls] = auc_score
    _t['mask'].toc()
    print('Im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                      .format(i + 1, num_images, _t['im_detect'].average_time,
                              _t['mask'].average_time))
  class_f1 = np.zeros(imdb.num_classes)
  class_auc = np.zeros(imdb.num_classes)
  for j in range(1, imdb.num_classes):
    cls_f1 = all_f1[:, j]
    f1_ind = np.where(cls_f1 > 0.)[0]
    cls_f1 = cls_f1[f1_ind]
    class_f1[j] = np.average(cls_f1)

    cls_auc = all_auc[:, j]
    auc_ind = np.where(cls_auc > 0)[0]
    cls_auc = cls_auc[auc_ind]
    class_auc[j] = np.average(cls_auc)

  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('Test Results:')
  print('Average F1  Score: %.3f' %np.average(class_f1[1:]))
  print('Average AUC Score: %.3f' %np.average(class_auc[1:]))
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('\n')
  print('============================================================')

  if os.path.isfile(os.path.join(output_dir, 'detections.pkl')):
    all_boxes = pickle.load(open(os.path.join(output_dir, 'detections.pkl'), 'rb'))
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    return
    print(np.array(all_boxes).shape)
    for i in range(np.array(all_boxes).shape[1]):
      test_boxes = [[[] for _ in range(1)] for _ in range(2)]
      test_boxes[1][0] = all_boxes[1][i]
      print(test_boxes)
      print(np.array(test_boxes).shape)
      imdb.evaluate_detections(test_boxes, output_dir)
    imdb.evaluate_detections(all_boxes, output_dir)
    return

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
 
 