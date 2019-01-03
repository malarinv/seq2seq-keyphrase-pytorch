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
        # parser = argparse.ArgumentParser(description='predictor',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # config.preprocess_opts(parser)
        # self.opt = parser.parse_args([])
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
        clean_src_str = self.preprocess_query(src_str)
        tokenized_test_pairs = pykp.io.get_tokenized_pairs(clean_src_str,self.model_opts,True)
        return pykp.io.process_dataset(tokenized_test_pairs,
                                           self.model_opts.word2id, self.model_opts.id2word,
                                           self.model_opts,
                                           dataset_name=test_dataset_name,
                                           data_type='test')

    def process(self,input_str,top_n=8):
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

        return output[:top_n]

    def removeSpecialChars(self, inputstring):
        import string
        translator = str.maketrans('', '', string.punctuation.replace('-',''))
        # inputstring = 'string with "punctuation" inside of it! Does this work? I hope so.'
        # print('hello', inputstring)
        inputstring = inputstring.decode('utf-8').translate(translator)
        return inputstring

    def preprocess_query(self, inputstring):
        inputstring = inputstring.lower().strip().encode('ascii',errors='ignore')
        inputstring = self.removeSpecialChars(inputstring)
        inputstring = self.removeWhiteSpace(inputstring)
        return inputstring

    def removeWhiteSpace(self, inputstring):
        outputstring = " ".join(inputstring.split())
        return outputstring


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

def main():
    import pandas as pd
    # import sys
    # sys.argv = 'python -data data_custom/kp20k/kp20k -vocab data_custom/kp20k/kp20k.vocab.pt -exp_path "./exp-custom/attn_general.input_feeding.copy/%s.%s" -train_from "exp-custom/attn_general.input_feeding.copy/kp20k-custom.ml.copy.20190102-100747/model/kp20k-custom.ml.copy.epoch=5.batch=40.total_batch=480.model" -model_path "./model/attn_general.input_feeding.copy/%s.%s" -pred_path "./pred-custom/attn_general.input_feeding.copy/%s.%s" -exp "kp20k-custom" -batch_size 256 -bidirectional -copy_attention -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode general -input_feeding -min_src_seq_length 5 -vocab_size 2107 -device_ids 0'.replace('"','').split(' ')
    kp = KeyphrasePredictor()
    data = pd.read_csv('./kbsearch.solrcore2.csv',encoding='utf-8')
    extract_products = lambda x:';'.join(kp.process(x))
    extract_products(data['question'].iloc[1])
    data['products'] = data['question'].apply(extract_products)
    data.to_csv('output.csv')

if __name__ == '__main__':
    main()
    # kp = KeyphrasePredictor()
    # kp.process('what would you charge me on pay order non-correspondent bank for privilege special scheme savings account')
