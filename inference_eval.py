#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:17:47 2019

@author: barry
"""
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, 
    load_scalar, get_labels)
from data_generator import DataGenerator
from models import TFSANN
from losses import binary_cross_entropy
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu
import config


def inference_eval(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: '1', which means using validation data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    iteration = args.iteration
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    seq_len = 640
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    labels = get_labels(taxonomy_level)
    classes_num = len(labels)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
        
    eval_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'evaluate.h5')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'evaluate.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 
        '{}_iterations.pth'.format(iteration))
    
    submission_path = os.path.join(workspace, 'submissions', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type, 'submission_evaluate.csv')
    create_folder(os.path.dirname(submission_path))
    
    annotation_path = os.path.join(dataset_dir, 'annotations.csv')
    
    yaml_path = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'taxonomy_level={}'.format(taxonomy_level), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)
        
    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num, seq_len, mel_bins, cuda)
    #model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        evaluate_hdf5_path=eval_hdf5_path, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        taxonomy_level=taxonomy_level, 
        cuda=cuda, 
        verbose=True)
    
    # Evaluate on validation data
    evaluator.evaluate(
        data_type='evaluate', 
        submission_path=submission_path, 
        annotation_path=annotation_path, 
        yaml_path=yaml_path, 
        max_iteration=None)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_inference_validation = subparsers.add_parser('inference_eval')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., TFSANN')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)
    
    if args.mode == 'inference_eval':
        inference_eval(args)

    else:
        raise Exception('Error argument!')