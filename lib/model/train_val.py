# --------------------------------------------------------
# Tensorflow RGB-N
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou , based on code from Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time
import pdb
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, sess, network, net, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    self.net_name = net
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indeces of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indeces of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def get_variables_in_checkpoint_file(self, file_name):
    try:
      print('load model')
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      print('load model')
      var_to_shape_map = reader.get_variable_to_shape_map()
      # for key in var_to_shape_map:
      #   print("tensor_name: ", key)
      # assert False
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def set_constrain(self):
    with tf.device("/gpu:0"):
      tmp_np = tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/weights:0').eval()
      tmp_np[2, 2, :, :] = 0
      for i in range(3):
        tmp_np[:, :, 0, i] = tmp_np[:, :, 0, i] / tmp_np[:, :, 0, i].sum()
        tmp_np[:, :, 1, i] = tmp_np[:, :, 1, i] / tmp_np[:, :, 1, i].sum()
        tmp_np[:, :, 2, i] = tmp_np[:, :, 2, i] / tmp_np[:, :, 2, i].sum()  # Element-wise division by the sum
      tmp_np[2, 2, :, :] = -1

      tmp_tensor = tf.convert_to_tensor(tmp_np,dtype=tf.float32)
      return tmp_tensor


  def train_model(self, sess, max_iters, loadfc, imdb_name):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Determine different scales for anchors, see paper
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      layers = self.net.create_architecture(sess, 'TRAIN', self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      loss = layers['total_loss']
      semi_loss = layers['ssl_contrastive_loss']
      # semi_loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      momentum = cfg.TRAIN.MOMENTUM
      self.optimizer = tf.train.MomentumOptimizer(lr, momentum)
      # self.semi_optimizer = tf.train.MomentumOptimizer(lr, momentum)

      if cfg.FREEZE:
        # 选择mask部分的参数
        train_var_list = [var for var in tf.trainable_variables() if 'mask_prediction' in var.name]
        train_var_name = [var.name for var in tf.trainable_variables() if 'mask_prediction' in var.name]
        print(train_var_name)
        gvs = self.optimizer.compute_gradients(loss,var_list=train_var_list) #自行选择优化器
      else:
        # Compute the gradients wrt the loss
        gvs = self.optimizer.compute_gradients(loss)
      semi_gvs = self.optimizer.compute_gradients(semi_loss)
      semi_train_op = self.optimizer.apply_gradients(semi_gvs)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((tf.clip_by_value(grad,-5.0,5.0), var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        #final_gvs = []
        #with tf.variable_scope('Gradient_Mult') as scope:
          #for grad, var in gvs:
            #final_gvs.append((tf.clip_by_value(grad,-50.0,50.0), var))
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

    # Find previous snapshots if there is any to restore from
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redstr = '_iter_{:d}.'.format(cfg.TRAIN.STEPSIZE[0]+1)
    sfiles = [ss.replace('.meta', '') for ss in sfiles]
    sfiles = [ss for ss in sfiles if redstr not in ss]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    nfiles = [nn for nn in nfiles if redstr not in nn]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    np_paths = nfiles
    ss_paths = sfiles

    if lsf == 0:
      if loadfc == 1:
        if cfg.FREEZE == False:
          print('Restorining model snapshots from {:s}'.format(self.pretrained_model))
          self.saver.restore(sess, str(self.pretrained_model))
          print('Restored.')
        else:
          # 选择encode部分参数
          no_train_var = [var for var in tf.global_variables() if 'mask_prediction' not in var.name]  #这里的'encode'是在设置网络过程中某个scope的命名
          # saver选择要导入的参数
          saver = tf.train.Saver(no_train_var)
          # 对整个网络所有参数做初始化
          init = tf.global_variables_initializer()
          sess.run(init)
          # encode部分参数覆盖
          saver.restore(sess, str(self.pretrained_model))
          print('Freeze Restored.')
      else:
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignorizing the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
        if len(variables_to_restore) > 0:
          restorer = tf.train.Saver(variables_to_restore)
          restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        if cfg.TRAIN.FUSE or 'coco' in imdb_name:
          print('noise variable replace')
          noise_vars = {}
          for v in variables:
            if 'res101' in self.net_name:
              if v.name.split('/')[0] == 'noise' and v.name.split(':')[0].replace('noise', 'resnet_v1_101', 1) in var_keep_dic:
                noise_vars[v.name.split(':')[0].replace('noise', 'resnet_v1_101', 1)] = v
            elif 'res50' in self.net_name:
              if v.name.split('/')[0] == 'noise' and v.name.split(':')[0].replace('noise', 'resnet_v1_50', 1) in var_keep_dic:
                noise_vars[v.name.split(':')[0].replace('noise', 'resnet_v1_50', 1)] = v
          # pdb.set_trace()
          noise_restorer = tf.train.Saver(noise_vars)
          noise_restorer.restore(sess, self.pretrained_model)
        elif cfg.USE_MASK is True and imdb_name !='dist_cover_train_single':
        # elif cfg.USE_MASK is True and imdb_name =='casia_train_all_single':
        # elif cfg.USE_MASK is True:
          mask_vars={}
          for v in variables:
            # print(v.name)
            if 'res101' in self.net_name:
              if len(v.name.split('/')) > 1 and v.name.split('/')[1]=='mask_conv' and v.name.split(':')[0].split('mask_prediction/')[1].replace('mask_conv','resnet_v1_101',1) in var_keep_dic:
              # if len(v.name.split('/')) > 1 and v.name.split('/')[1]=='mask_conv' and v.name.split(':')[0].split('mask_prediction/')[1].replace('mask_conv','noise',1) in var_keep_dic:
                mask_vars[v.name.split(':')[0].split('mask_prediction/')[1].replace('mask_conv','resnet_v1_101',1)]=v
                # mask_vars[v.name.split(':')[0].split('mask_prediction/')[1].replace('mask_conv','noise',1)]=v

          #pdb.set_trace()
          print('mask conv len',len(mask_vars))
          mask_restorer = tf.train.Saver(mask_vars)
          mask_restorer.restore(sess, self.pretrained_model)
          print('Mask Loaded')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')

      trainable_variables = tf.trainable_variables()
      sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
      last_snapshot_iter = 0
    else:
      # Get the most recent snapshot and restore
      ss_paths = [ss_paths[-1]]
      np_paths = [np_paths[-1]]

      print('Restorining model snapshots from {:s}'.format(sfiles[-1]))
      self.saver.restore(sess, str(sfiles[-1]))
      print('Restored.')
      # Needs to restore the other hyperparameters/states for training, (TODO xinlei) I have
      # tried my best to find the random states so that it can be recovered exactly
      # However the Tensorflow state is currently not available
      with open(str(nfiles[-1]), 'rb') as fid:
        st0 = pickle.load(fid)
        cur = pickle.load(fid)
        perm = pickle.load(fid)
        cur_val = pickle.load(fid)
        perm_val = pickle.load(fid)
        last_snapshot_iter = pickle.load(fid)

        np.random.set_state(st0)
        self.data_layer._cur = cur
        self.data_layer._perm = perm
        self.data_layer_val._cur = cur_val
        self.data_layer_val._perm = perm_val

        # Set the learning rate, only reduce once
        if last_snapshot_iter > cfg.TRAIN.STEPSIZE[0]:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
        # elif last_snapshot_iter // cfg.TRAIN.STEPSIZE == 2:
        #   sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA * cfg.TRAIN.GAMMA))
        else:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()

    if 'constrained' in self.net_name or 'cbam' in self.net_name:
      # reset constrain_conv
      print('reset constrained')

      tmp_tensor1 = self.set_constrain()
      update_weights1 = tf.assign(tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/weights:0'),
                                  tmp_tensor1)
      biase11 = tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/biases:0')
      biase1 = tf.multiply(biase11, 0)

      update_biases1 = tf.assign(tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/biases:0'), biase1)
      update1 = [update_weights1, update_biases1]
      sess.run(update1)

      # ini op used in while
      with tf.control_dependencies([train_op]):
        tmp_tensor = self.set_constrain()
        update_weights = tf.assign(tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/weights:0'),
                                   tmp_tensor)

      biase1 = tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/biases:0')
      biase = tf.multiply(biase1, 0)
      with tf.control_dependencies([update_weights]):
        update_biases = tf.assign(tf.get_default_graph().get_tensor_by_name('noise/constrained_conv/biases:0'), biase)


    while iter < max_iters + 1:
      # Learning rate
      if iter == cfg.TRAIN.STEPSIZE[0] + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
      elif iter == cfg.TRAIN.STEPSIZE[1] + 1 :
        self.snapshot(sess, iter)
        sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()

      now = time.time()
      if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        if cfg.USE_MASK is True:
          rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, total_loss, contrastive_loss, summary = \
              self.net.train_step_with_summary_with_mask(sess, update_weights,update_biases, blobs, train_op)
        else:
          if 'constrained' in self.net_name or 'cbam' in self.net_name:
            if blobs['gt_boxes'][0, 4] == -1:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary, contrastive_loss = \
                self.net.train_semi_step_with_summary(sess, update_weights, update_biases, blobs, semi_train_op, iter)
            else:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary, contrastive_loss = \
                self.net.train_step_with_summary(sess, update_weights, update_biases, blobs, train_op, iter)
          else:
            if blobs['gt_boxes'][0, 4] == -1:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary, contrastive_loss = \
                self.net.train_semi_step_with_summary(sess, blobs, semi_train_op, iter)
            else:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary, contrastive_loss = \
                self.net.train_step_with_summary(sess, blobs, train_op, iter)

        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val, iter)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        if cfg.USE_MASK is True:
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, total_loss, contrastive_loss = \
              self.net.train_step_with_mask(sess, update_weights,update_biases, blobs, train_op)
        else:
          if 'constrained' in self.net_name or 'cbam' in self.net_name:
            if blobs['gt_boxes'][0, 4] == -1:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, contrastive_loss = \
                self.net.train_semi_step(sess, update_weights, update_biases, blobs, semi_train_op, iter)
            else:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, contrastive_loss = \
                self.net.train_step(sess, update_weights, update_biases, blobs, train_op, iter)
          else:
            if blobs['gt_boxes'][0, 4] == -1:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, contrastive_loss = \
                self.net.train_semi_step(sess, blobs, semi_train_op, iter)
            else:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, contrastive_loss = \
                self.net.train_step(sess, blobs, train_op, iter)
      timer.toc()

      # Display training information
      # if iter % 1 == 0:
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        if cfg.USE_MASK is True:
            print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                  '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> contrastive_loss: %.6f\n >>> loss_mask: %.6f\n >>> lr: %f' % \
                  (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, contrastive_loss, loss_mask, lr.eval()))
            print('speed: {:.3f}s / iter'.format(timer.average_time))
            print('remaining time: {:.3f}h'.format(((max_iters - iter) * timer.average_time) / 3600))
        else:
            print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                  '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> contrastive_loss: %.6f\n >>> lr: %f' % \
                  (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, contrastive_loss, lr.eval()))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        snapshot_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(snapshot_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
          for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        if len(ss_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
          for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
              os.remove(str(sfile))
            else:
              os.remove(str(sfile + '.data-00000-of-00001'))
              os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')
  if cfg.TRAIN.USE_NOISE_AUG:
    print('Appending noise to training examples...')
    imdb.append_noise_images()
    print('done')
  if cfg.TRAIN.USE_JPG_AUG:
    print('Appending jpg compression to training examples...')
    imdb.append_jpg_images()
    print('done')
  if cfg.TRAIN.USE_COLOR_AUG:
    print('Appending color jitter to training examples...')
    imdb.append_colorjitter_images()
    print('done')


  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                        (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000, loadfc=0,imdb_name=None):
  """Train a Fast R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, net, imdb, roidb, valroidb, output_dir, tb_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters, loadfc, imdb_name)
    print('done solving')