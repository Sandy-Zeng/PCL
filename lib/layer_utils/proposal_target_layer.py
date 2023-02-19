# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
import os
from model.nms_wrapper import nms

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # print(gt_boxes[:, -1])
  if gt_boxes[0, -1] != -1:
    # print('proposal target layers')
    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.USE_GT:
      zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
      all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
      )
      # not sure if it a wise appending, but anyway i am not using it
      all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
  else:
    rois, roi_scores, labels = _unlabeled_proposal_target_layer(rpn_rois, rpn_scores)
    labels = labels.reshape(-1, 1).astype(np.float32)
    bbox_targets = np.zeros((labels.shape[0], _num_classes * 4), dtype=gt_boxes.dtype)
    bbox_inside_weights = np.zeros((labels.shape[0], _num_classes * 4), dtype=gt_boxes.dtype)
    bbox_outside_weights = np.zeros((labels.shape[0], _num_classes * 4), dtype=gt_boxes.dtype)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


# def _unlabeled_sample_rois(all_rois, all_scores, fg_rois_per_image, rois_per_image):
#   topscore_inds = np.argsort(all_scores.reshape(-1))[::-1]
#   fg_inds = topscore_inds[:int(fg_rois_per_image)]
#
#   all_scores_reshape = np.reshape(all_scores, (-1))
#   bg_inds = np.where((all_scores_reshape > 0.1) & (all_scores_reshape < 0.5))[0]
#   if bg_inds.size == 0:
#     inds = topscore_inds.size - (int(rois_per_image - fg_rois_per_image)+100)
#     bg_inds = topscore_inds[inds:].reshape(-1)
#   to_replace = bg_inds.size < int(rois_per_image - fg_rois_per_image)
#   # print(bg_inds.shape)
#   bg_inds = npr.choice(bg_inds, size=int(rois_per_image - fg_rois_per_image), replace=to_replace)
#
#   # The indices that we're selecting (both fg and bg)
#   keep_inds = np.append(fg_inds, bg_inds)
#   rois = all_rois[keep_inds]
#   roi_scores = all_scores[keep_inds]
#   labels = np.array([1] * len(fg_inds) + [0] * len(bg_inds))
#   return labels, rois, roi_scores


def _unlabeled_sample_rois(all_rois, all_scores, fg_rois_per_image, rois_per_image):
  # Non-maximal suppression
  proposal_num = int(rois_per_image)
  rois = all_rois[:proposal_num, :]
  roi_scores = all_scores[:proposal_num, :]

  # print(keep_inds)
  labels = np.array([1] * int(fg_rois_per_image) + [0] * int(proposal_num-fg_rois_per_image))
  return labels, rois, roi_scores


def _unlabeled_proposal_target_layer(rpn_rois, rpn_scores):
  # print('unlabeled proposal layers')
  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores = _unlabeled_sample_rois(all_rois, all_scores, fg_rois_per_image, rois_per_image)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  return rois, roi_scores, labels

def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  if num_classes<=2:
    inds = np.where(clss > 0)[0]
  else:
    inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  gt_assignment = overlaps.argmax(axis=1)
  max_overlaps = overlaps.max(axis=1)
  labels = gt_boxes[gt_assignment, 4]

  # print(all_scores.shape)
  # topscore_inds = np.argsort(all_scores.reshape(-1))[::-1]
  # print(topscore_inds)
  # for id in topscore_inds:
  #   print('id', id)
  #   item_overlaps = bbox_overlaps(
  #     np.ascontiguousarray(all_rois[[id], 1:5], dtype=np.float),
  #     np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float))
  #   print(np.sort(item_overlaps)[::-1])
  #   print(np.max(item_overlaps))
  #   assert False
  # assert False

  # print(overlaps.shape)
  # print(all_scores.shape)
  #
  # if os.path.exists('./dis_figure') == False:
  #   os.mkdir('./dis_figure')
  # plt.figure()  # 创建图表1
  # # plt.ylim(0.7, 1)
  # plt.title('iou/cls_scores')  # give plot a title
  # plt.xlabel('cls scores')  # make axis labels
  # plt.ylabel('iou')
  # plt.scatter(all_scores.reshape(-1), overlaps.reshape(-1))
  # plt.grid()
  # inds = random.randint(0, 100)
  # plt.savefig('./dis_figure/%d_dis.png' % (inds))

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # print(fg_inds.size, bg_inds.size)
  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # print('proposal label', fg_inds)
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]
  # print(range(int(fg_rois_per_image)))

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
