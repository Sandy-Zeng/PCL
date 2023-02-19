# --------------------------------------------------------
# Tensorflow RGB-N
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.coco import coco
from datasets.casia import casia
from datasets.dist_fake import dist_fake
from datasets.nist import nist
from datasets.dvmm import dvmm
from datasets.columbia import columbia
from datasets.imd import imd 
import numpy as np


# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split> 
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'COCO_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

dvmm_path='/data/Columbia'
for split in ['dist_train', 'dist_test', 'dist_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: dvmm(split,2007,dvmm_path))
    
columbia_path='/data/Columbia/probe'
for split in ['dist_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: columbia(split,2007,columbia_path))

dso_path='/data/COVERAGE/probe'
for split in ['dist_cover_train_single', 'dist_cover_test_single']:
    name = split
    __sets[name] = (lambda split=split: dist_fake(split, 2007, dso_path))

nist_path='/data/NIST2016'
for split in ['dist_NIST_train_new_2', 'dist_NIST_train_new_2_0.5', 'dist_NIST_train_new_2_semi', 'dist_NIST_test_new_2']:
    name = split
    __sets[name] = (lambda split=split: nist(split,2007,nist_path))

nist_path='/data/NIST2016'
for split in ['dist_NIST_train_new_2_semi', 'dist_NIST_test_new_2']:
    name = split
    __sets[name] = (lambda split=split: nist(split,2007,nist_path))

casia_path='/data/CASIA/CASIA2/Tp'
for split in ['casia_train_all_single', 'casia_train_all_single_0.5', 'casia_train_all_single_semi']:
    name = split
    __sets[name] = (lambda split=split: casia(split,2007,casia_path))

casia1_path='/data/CASIA/CASIA1/Sp'
for split in ['casia_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: casia(split,2007,casia1_path))

coco_single_path = '/data/coco_synthetic/train_data'
for split in ['coco_train_filter_single','coco_test_filter_single', 'coco_train_filter', 'coco_test_filter']:
    name = split
    __sets[name] = (lambda split=split: coco(split, 2007, coco_single_path))

for split in ['coco_train_filter_single_0.5', 'coco_train_filter_single_semi']:
    name = split
    __sets[name] = (lambda split=split: coco(split, 2007, coco_single_path))

imd2020_path = '/data/IMD2020'
for split in ['imd2020_test_all_tamper', 'imd2020_train_split_tamper', 'imd2020_test_split_tamper']:
    name = split
    __sets[name] = (lambda split=split: imd(split, 2007, imd2020_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
