#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from evaluate import evaluate_beam_search,predict_beam_search
import logging
import numpy as np

import config
import utils

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from pykp.dataloader import KeyphraseDataLoader
from train import load_data_vocab, init_model, init_optimizer_criterion
from utils import Progbar, plot_learning_curve_and_write_csv
import pykp.io
import pykp
from pykp.io import KeyphraseDatasetTorchText, KeyphraseDataset

__author__ = "Malar Invention"
__email__ = "malarkannan.invention@gmail.com"

logger = logging.getLogger()


def generate_dataset():
    test_dataset_name = 'kp20k'
    src_fields = ['title', 'abstract']
    trg_fields = ['keyword']
    parser = argparse.ArgumentParser(
        description='preprocess_testset.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # **Preprocess Options**
    parser.add_argument('-source_dataset_root_dir', default='test/',
                        help="The path to the source data (raw json).")

    parser.add_argument('-output_path_prefix', default='data/',
                        help="Output file for the prepared data")

    config.preprocess_opts(parser)
    opt = parser.parse_args([])

    print("Loading Vocab...")
    opt.vocab_path = os.path.join(opt.output_path_prefix, 'kp20k', 'kp20k.vocab.pt')
    print(os.path.abspath(opt.vocab_path))
    word2id, id2word, vocab = torch.load(opt.vocab_path, 'rb')
    print('Vocab size = %d' % len(vocab))

    # for test_dataset_name in test_dataset_names:
    opt.source_test_file = os.path.join(opt.source_dataset_root_dir, '%s_testing.json' % (test_dataset_name))

    # output path for exporting the processed dataset
    opt.output_path = os.path.join(opt.output_path_prefix, test_dataset_name)
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    print("Loading test data...")

    tokenized_test_pairs = pykp.io.load_src_trgs_pairs(source_json_path=opt.source_test_file,
                                                       dataset_name=test_dataset_name,
                                                       src_fields=src_fields,
                                                       trg_fields=trg_fields,
                                                       valid_check=True,
                                                       opt=opt)

    print("Exporting complete dataset")

    # pykp.io.process_and_export_dataset(tokenized_test_pairs,
    #                                    word2id, id2word,
    #                                    opt,
    #                                    opt.output_path,
    #                                    dataset_name=test_dataset_name,
    #                                    data_type='test')
    return pykp.io.process_dataset(tokenized_test_pairs,
                                       word2id, id2word,
                                       opt,
                                       opt.output_path,
                                       dataset_name=test_dataset_name,
                                       data_type='test')


class KeyphrasePredictor(object):
    """docstring for KeyphrasePredictor."""
    def __init__(self):
        super(KeyphrasePredictor, self).__init__()
        self.model_opts = config.init_opt(description='predictor')
        # self.vocab_path = self.model_opts.vocab#os.path.join(self.model_opts.data, 'kp20k', 'kp20k.vocab.pt')
        parser = argparse.ArgumentParser(description='predictor',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        config.preprocess_opts(parser)
        self.opt = parser.parse_args([])
        self.load()

    def load(self):
        word2id, id2word, vocab = torch.load(self.model_opts.vocab, 'rb')
        self.model_opts.word2id = word2id
        self.model_opts.id2word = id2word
        self.model_opts.vocab = vocab
        self.model = init_model(self.model_opts)
        self.generator = SequenceGenerator(self.model,
                                      eos_id=self.model_opts.word2id[pykp.io.EOS_WORD],
                                      beam_size=self.model_opts.beam_size,
                                      max_sequence_length=self.model_opts.max_sent_length
                                      )



    def preprocess_input(self,src_str):
        test_dataset_name='kp20k'
        tokenized_test_pairs = pykp.io.get_tokenized_pairs(src_str,self.opt,True)
        return pykp.io.process_dataset(tokenized_test_pairs,
                                           self.model_opts.word2id, self.model_opts.id2word,
                                           self.opt,
                                           dataset_name=test_dataset_name,
                                           data_type='test')

    def process(self,input_str):
        one2one,one2many = self.preprocess_input(input_str)
        # test_data_loaders, word2id, id2word, vocab = load_vocab_and_testsets(self.opt,one2one,one2many)
        pin_memory = torch.cuda.is_available()
        testset_name = 'kp20k'
        logger.info("Loading test dataset %s" % testset_name)
        # testset_path = os.path.join(opt.test_dataset_root_path, testset_name, testset_name + '.test.one2many.pt')
        # test_one2many = torch.load(testset_path, 'wb')
        test_one2many_dataset = KeyphraseDataset(one2many, word2id=self.model_opts.word2id, id2word=self.model_opts.id2word, type='one2many', include_original=True)
        test_one2many_loader = KeyphraseDataLoader(dataset=test_one2many_dataset,
                                                   collate_fn=test_one2many_dataset.collate_fn_one2many,
                                                   num_workers=self.model_opts.batch_workers,
                                                   max_batch_example=self.model_opts.beam_search_batch_example,
                                                   max_batch_pair=self.model_opts.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)
        # test_one2many_loaders = [test_one2many_loader]
        # for testset_name, test_data_loader in zip(['kp20k'], test_one2many_loaders):
        # test_data_loader = test_one2many_loader
        logger.info('Evaluating %s' % testset_name)
        output = predict_beam_search(self.generator, test_one2many_loader, self.model_opts,
                             title='test_%s' % testset_name,
                             predict_save_path=None)#opt.pred_path + '/%s_test_result/' % (testset_name))
        print(output)


def load_vocab_and_testsets(opt,test_one2one,test_one2many):
    logger.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab, 'rb')
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    logger.info('#(vocab)=%d' % len(vocab))
    logger.info('#(vocab used)=%d' % opt.vocab_size)

    pin_memory = torch.cuda.is_available()
    test_one2many_loaders = []

    for testset_name in ['kp20k']:
        logger.info("Loading test dataset %s" % testset_name)
        # testset_path = os.path.join(opt.test_dataset_root_path, testset_name, testset_name + '.test.one2many.pt')
        # test_one2many = torch.load(testset_path, 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
        test_one2many_loader = KeyphraseDataLoader(dataset=test_one2many_dataset,
                                                   collate_fn=test_one2many_dataset.collate_fn_one2many,
                                                   num_workers=opt.batch_workers,
                                                   max_batch_example=opt.beam_search_batch_example,
                                                   max_batch_pair=opt.beam_search_batch_size,
                                                   pin_memory=pin_memory,
                                                   shuffle=False)

        test_one2many_loaders.append(test_one2many_loader)
        logger.info('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' % (len(test_one2many_loader.dataset), test_one2many_loader.one2one_number(), len(test_one2many_loader)))
        logger.info('*' * 50)

    return test_one2many_loaders, word2id, id2word, vocab


def main():
    opt = config.init_opt(description='predict_keyphrase.py')
    logger = config.init_logging('predict_keyphrase', opt.exp_path + '/output.log', redirect_to_stdout=False)

    logger.info('EXP_PATH : ' + opt.exp_path)

    logger.info('Parameters:')
    [logger.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    logger.info('======================  Checking GPU Availability  =========================')
    if torch.cuda.is_available():
        if isinstance(opt.device_ids, int):
            opt.device_ids = [opt.device_ids]
        logger.info('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opt.device_ids) > 1 else '1 GPU', str(opt.device_ids)))
    else:
        logger.info('Running on CPU!')

    try:
        one2one,one2many = generate_dataset()
        test_data_loaders, word2id, id2word, vocab = load_vocab_and_testsets(opt,one2one,one2many)
        model = init_model(opt)
        generator = SequenceGenerator(model,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length
                                      )

        for testset_name, test_data_loader in zip(['kp20k'], test_data_loaders):
            logger.info('Evaluating %s' % testset_name)
            output = predict_beam_search(generator, test_data_loader, opt,
                                 title='test_%s' % testset_name,
                                 predict_save_path=None)#opt.pred_path + '/%s_test_result/' % (testset_name))
            print(output)
    except Exception as e:
        logger.error(e, exc_info=True)

'''
python -m predict_keyphrase -data data/kp20k/kp20k -vocab data/kp20k/kp20k.vocab.pt -exp_path "./exp-1/attn_general.input_feeding.copy/%s.%s" -model_path "./model-1/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-1/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-1" -batch_size 256 -bidirectional -copy_attention -run_valid_every -1 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding
'''
if __name__ == '__main__':
    kp = KeyphrasePredictor()
    kp.process('what would you charge me on pay order non-correspondent bank for privilege special scheme savings account')
