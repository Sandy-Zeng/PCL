# --------------------------------------------------------
# Tensorflow RGB-N
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np

from nets.network_cbam import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
# import sys
# sys.path.append('../')
# print(sys.path)
from compact_bilinear_pooling.compact_bilinear_pooling import compact_bilinear_pooling_layer
import pdb

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.RESNET.BN_TRAIN,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnet_cbam(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # Get the normalized coordinates of bboxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / np.float32(self._feat_stride[0])
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / np.float32(self._feat_stride[0])
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / np.float32(self._feat_stride[0])
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / np.float32(self._feat_stride[0])
        # Won't be back-propagated to rois anyway, but to save time
        if cfg.RESNET.MAX_POOL:
            pre_pool_size = cfg.POOLING_SIZE * 2
            spacing_w = (x2 - x1) / pre_pool_size
            spacing_h = (y2 - y1) / pre_pool_size
            x1 = (x1 + spacing_w / 2) / (tf.to_float(bottom_shape[2]) - 1.)
            y1 = (y1 + spacing_h / 2) / (tf.to_float(bottom_shape[1]) - 1.)
            nw = spacing_w * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[2]) - 1.)
            nh = spacing_h * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[1]) - 1.)
            x2 = x1 + nw
            y2 = y1 + nh
            bboxes = tf.stop_gradient(tf.concat(1, [y1, x1, y2, x2]))

            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")
            crops = slim.avg_pool2d(crops, [2, 2], [2, 2], padding='SAME')
            #        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
        else:
            pre_pool_size = cfg.POOLING_SIZE
            spacing_w = (x2 - x1) / pre_pool_size
            spacing_h = (y2 - y1) / pre_pool_size
            x1 = (x1 + spacing_w / 2) / (tf.to_float(bottom_shape[2]) - 1.)
            y1 = (y1 + spacing_h / 2) / (tf.to_float(bottom_shape[1]) - 1.)
            nw = spacing_w * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[2]) - 1.)
            nh = spacing_h * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[1]) - 1.)
            x2 = x1 + nw
            y2 = y1 + nh

            bboxes = tf.stop_gradient(tf.concat(1, [y1, x1, y2, x2]))
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],name="crops")
        return crops

  def cbam_module(self,inputs, reduction_ratio=0.5, kernel=7, name=''):
      with tf.variable_scope("cbam_" + name, reuse=None):

          batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]

          maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keep_dims=True), axis=2, keep_dims=True)
          avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keep_dims=True), axis=2, keep_dims=True)

          maxpool_channel = slim.flatten(maxpool_channel)
          avgpool_channel = slim.flatten(avgpool_channel)

          mlp_1_max = slim.fully_connected(inputs=maxpool_channel, num_outputs=int(hidden_num * reduction_ratio), scope="mlp_1",
                                      reuse=None)

          mlp_2_max = slim.fully_connected(inputs=mlp_1_max, num_outputs=hidden_num, scope="mlp_2", reuse=None)
          mlp_2_max = tf.reshape(mlp_2_max, [-1, 1, 1, hidden_num])

          mlp_1_avg = slim.fully_connected(inputs=avgpool_channel, num_outputs=int(hidden_num * reduction_ratio), scope="mlp_1",
                                      reuse=True)
          mlp_2_avg = slim.fully_connected(inputs=mlp_1_avg, num_outputs=hidden_num, scope="mlp_2", reuse=True)
          mlp_2_avg = tf.reshape(mlp_2_avg, [-1, 1, 1, hidden_num])

          channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
          # channel_refined_feature = inputs * channel_attention

          maxpool_spatial = tf.reduce_max(inputs, axis=3, keep_dims=True)
          avgpool_spatial = tf.reduce_mean(inputs, axis=3, keep_dims=True)
          max_avg_pool_spatial = tf.concat(3,[maxpool_spatial, avgpool_spatial])
          conv_layer = slim.conv2d(max_avg_pool_spatial, 1, [kernel, kernel], padding="SAME",
                                   activation_fn=None,scope='conv_layer')
          spatial_attention = tf.nn.sigmoid(conv_layer)

          # refined_feature = channel_refined_feature * spatial_attention
      
      return channel_attention, spatial_attention
      # return refined_feature

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self, image, scope):
    with tf.variable_scope(scope):
      net = resnet_utils.conv2d_same(image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      # initializer = tf.contrib.layers.xavier_initializer()
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 101:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 152:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    with tf.variable_scope('noise'):
      conv=slim.conv2d(self.noise, num_outputs=3, kernel_size=[5,5], stride=1 , padding='SAME', activation_fn=None, trainable=is_training, scope='constrained_conv')
    self._layers['noise']=conv
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C_1 = self.build_base(conv, 'noise')
        C_2, _ = resnet_v1.resnet_v1(C_1,
                                     blocks[0:1],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope='noise')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C_3, _ = resnet_v1.resnet_v1(C_2,
                                     blocks[1:2],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope='noise')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C_4, end_point = resnet_v1.resnet_v1(C_3,
                                     blocks[2:3],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope='noise')
    self.end_point=end_point
    self._act_summaries.append(C_4)
    self._layers['head'] = C_4

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()
      rpn1 = slim.conv2d(C_4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      c_atten, s_atten = self.cbam_module(inputs=rpn1, name="rpn_conv1")
      rpn = rpn1 * c_atten * s_atten
      # rpn = rpn1

      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')

      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")


      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      if cfg.POOLING_MODE == 'crop':
        noise_pool5 = self._crop_pool_layer(C_4, rois, "constrained_pool5")
      else:
        raise NotImplementedError

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        fc7, end_point1 = resnet_v1.resnet_v1(noise_pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope='noise')
    self._layers['fc7']=fc7
    self.end_point1=end_point1

    with tf.variable_scope('noise_pred'):
    # with tf.variable_scope(self._resnet_scope):
      cls_fc7 = tf.reduce_mean(fc7, axis=[1, 2])

      cls_score = slim.fully_connected(cls_fc7, self._num_classes, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      box_fc7=tf.reduce_mean(fc7, axis=[1, 2])
      bbox_pred = slim.fully_connected(box_fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      elif v.name == ('noise' + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      elif v.name.split('/')[0]=='noise_pred':
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)
      # else:
        # print('Varibles not restored: %s' % v.name)
    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('not Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        # restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc = tf.train.Saver({'noise' + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        # sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                          #  tf.reverse(conv1_rgb, [False,False,True,False])))
        
        sess.run(tf.assign(self._variables_to_fix['noise' + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [False,False,True,False])))
