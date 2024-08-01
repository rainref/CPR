# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import os.path as osp
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import component.config as config
import component.inputters.utils as util

from collections import OrderedDict, Counter
from tqdm import tqdm
from component.inputters.timer import AverageMeter, Timer
import component.inputters.vector as vector
import component.inputters.dataset as data

from model import CCSGModel
from component.eval.bleu import corpus_bleu
from component.eval.rouge import Rouge
from component.eval.meteor import Meteor

import pickle
from torch.utils.tensorboard import SummaryWriter
from component.eval import metrics

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--fuse_type', type=str, default='multi_attn', choices=['multi_attn', 'sum', 'cat'])
    runtime.add_argument('--debug', action="store_true")
    runtime.add_argument('--debug_data_len', type=int, default=100)
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--eval_epochs', type=int, default=1,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')

    files.add_argument('--train_dataset', nargs='+',type=str,  default="/root/CCSG/data/data_bpe/train_java_construct_entire.pkl",
                       help='the dir of dataset')
    files.add_argument('--test_dataset', type=str,  default="/root/CCSG/data/data_bpe/eval_java_construct_entire.pkl",
                       help='the dir of dataset')
    files.add_argument('--pretrain_dataset', type=str,  default="/root/CCSG/data/data_bpe/pretrain_entire.pkl",
                       help='the dir of dataset')
    files.add_argument('--dicts', type=str,  default="/root/CCSG/data/data_bpe/dicts.pkl.lower",
                       help='the dir of dictionary')




    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')


    files.add_argument('--bpe_vocab', nargs='+', type=str, default='bpe_vocab.out',
                       help='bpe vocabulary')

    files.add_argument('--train_cache', nargs='+', type=str,
                       help='the cache file of processed graphs')




    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and target words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=50000,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')
    preprocess.add_argument('--node_type_tag', type='bool', default=True,
                            help='involve type tags of Graph into Embedding')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=False,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')
    general.add_argument('--micro_change', type='bool', default=False,
                         help='Micro change mode')
    general.add_argument('--use_cuda', type='bool', default=False,
                         help='whether use cuda')
    general.add_argument('--use_bpe', type='bool', default=False,
                         help='whether use BPE')
    general.add_argument('--tqdm_miniter', type=int, default=1,
                         help='miniter of tqdm')
    general.add_argument('--submodel_dim', type=int, default=512,
                         help='the dimension of token quantity regressor')
    general.add_argument('--MTL', type='bool', default=True,
                         help='whether use MTL to fit the token quantity')
    general.add_argument("--virtual",type="bool",default=False,
                         help='whether use virtual nodes'
                         )
    general.add_argument("--constant_weight", nargs='+', type=float, default=None, help="the weight of loss")
    general.add_argument("--singleToken", type='bool', default=False,
                         help='activate this attribute to swift to single token')
    general.add_argument('--use_seq', type='bool', default=True,
                         help='whether use seq')
    general.add_argument('--use_gnn', type='bool', default=True,
                         help='whether to use gnn')
    general.add_argument('--template', type='bool', default=True,
                         help='template prediction method')
    general.add_argument('--temp_rate', type=float, default=0.3,
                         help='the loss rate of temp prediction')
    general.add_argument('--repeat_mode', type="bool",default=False,
                         help='repeat in the other decoder')
    general.add_argument('--STL-learning', type="bool",default=False,
                         help='repeat in the other decoder')
    general.add_argument('--fine_tune_template', type="bool", default=False,
                         help='repeat in the other decoder')
    general.add_argument('--pretrain_stage', type="bool", default=False,
                         help='repeat in the other decoder')
    general.add_argument('--from_pretrain', type=str,  default="/root/CCSG/main/java_base/pretrain_model/pretrain_model.mdl",
                       help='the dir of dataset')
    general.add_argument('--no_graph_rate', type=float, default=0.01,
                         help='the loss rate of temp prediction')

    general.add_argument('--no_ref', type='bool', default=False,
                     help='Print only one target sequence')
    general.add_argument('--no_field_emb', type='bool', default=False,
                     help='Print only one target sequence')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    suffix = '_mc_test' if args.micro_change else suffix
    model_dir = osp.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    args.model_file = osp.join(model_dir, 'model.mdl')
    if args.pretrain_stage:
        args.model_file=osp.join(model_dir, 'pretrain_model.mdl')
    args.log_file = osp.join(model_dir, 'model' + suffix + '.txt')
    args.pred_file = osp.join(model_dir, 'model' + suffix + '.json')
    if args.pretrained:
        args.pretrained = osp.join(model_dir, 'model.mdl')

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, vocabs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    if os.path.isfile(args.from_pretrain):
        pretrain_model=torch.load(open(args.from_pretrain,mode="rb"), map_location=lambda storage, loc: storage)
        vocabs = pretrain_model["vocab_attr"],pretrain_model["vocab_type"],\
                 pretrain_model["vocab_token"],pretrain_model["vocab_sep"],pretrain_model["vocab_identi"]
        state_dict=pretrain_model['state_dict']
        logger.info("init from pretrain model: {}".format(args.from_pretrain))
    else:
        state_dict=None

    model = CCSGModel(config.get_model_args(args), vocabs,state_dict)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, tb_writer: SummaryWriter = None):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader, ncols=150)

    pbar.set_description("%s" % 'Epoch = %d [perplexity = xxx, ml_loss = xxx,loss1 = xxx loss2 = xxx]' %
                         current_epoch)
    if args.STL_learning:
        T_all = global_stats['T_all']
        cut = args.STLR_cut * T_all
        ratio=args.STLR_ratio

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        if args.STL_learning:
            t = model.updates + 1
            if t <= T_all:
                if t < cut:
                    p=t/cut
                else:
                    p=1-((t-cut)/(cut * (1/args.STLR_cut-1)))
                nt=args.STLR_max*((1+p*(ratio-1))/ratio)
            else:
                nt=args.STLR_starting
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = nt

        else:
            if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
                cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = cur_lrate

        net_loss = model.update(ex,current_epoch)
        if net_loss['ml_loss']!=-1:
            old_avg=ml_loss.avg
            ml_loss.update(net_loss['ml_loss'], bsz)
            new_avg=ml_loss.avg
            perplexity.update(net_loss['perplexity'], bsz)

        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.6f]' % \
                       (current_epoch, perplexity.avg, ml_loss.avg)

        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.6f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))
    tb_writer.add_scalar('train/ml_loss', ml_loss.avg, current_epoch)
    tb_writer.add_scalar('train/perplexity', perplexity.avg, current_epoch)
    # Checkpoint

    model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev', tb_writer: SummaryWriter = None):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    right_tp=0
    sources, hypotheses, references, copy_dict,types,hyp_tps,ref_tps = dict(), dict(), dict(), dict(),dict(),dict(),dict()
    sum_no_graph_number=0
    with torch.no_grad():
        pbar = tqdm(data_loader)
        ending_idx=0
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ex_ids = list(range(ending_idx, ending_idx+batch_size))
            ending_idx=ending_idx+batch_size
            if not args.micro_change:
                template_prediction,identi_prediction,text,targets,raw_template,no_graph_number = model.eval(ex)
                sum_no_graph_number=sum_no_graph_number+no_graph_number
            else:
                template_prediction, identi_prediction, text, targets,raw_template = model.predict(ex)

            targets= [[str(i.value) for i in t] for t in targets] if not isinstance(targets[0][0],str) else targets
            src_sequences = [code for code in ex['raw_text']]
            # type_sequence=[type for type in ex['eval_types']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, text, targets):
                # if args.use_bpe:
                #     pred = pred.replace("@@ ?$", "").replace("@@ ", "")
                # if args.singleToken:
                #     pred = pred.split(" ")[0]

                hypotheses[key] = pred
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src



            # if copy_info is not None:
            #     copy_info = copy_info.cpu().numpy().astype(int).tolist()
            #     for key, cp in zip(ex_ids, copy_info):
            #         copy_dict[key] = cp

            pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    fp = open(args.pred_file, 'w')
    for hyp,ref,src in zip(hypotheses.values(),references.values(),sources.values()):
        code_str=src
        res_dict={"code":code_str,"ref":" ".join(ref),"hyp":" ".join(hyp)}
        string=json.dumps(res_dict)
        print(string, file=fp)
    fp.close()

    cur_epoch = global_stats['epoch']
    copy_dict = None if len(copy_dict) == 0 else copy_dict
    # bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
    #                                                                references,
    #                                                                copy_dict,
    #                                                                sources=sources,
    #                                                                filename=args.pred_file,
    #                                                                print_copy_info=args.print_copy_info,
    #                                                                mode=mode,eval_types=types)
    bleu1,bleu2,bleu3,bleu4,lv,perfect=eval_official_scores(hypotheses,
               references,
               copy_dict,
               sources=sources,
               filename=args.pred_file)

    bleu1, bleu2, bleu3, bleu4, lv, perfect=bleu1*100,bleu2*100,bleu3*100,bleu4*100,lv*100,perfect*100

    # def is_perfect_prediction(a, b):
    #     a = ''.join(a)
    #     b = ''.join(b)
    #     a = a.split()
    #     b = b.split()
    #     a = ''.join(a)
    #     b = ''.join(b)
    #     return a == b
    #
    # for key in references.keys():
    #     if is_perfect_prediction(hypotheses[key], references[key]):
    #         pred_correct += 1

    tb_writer.add_scalar('eval/bleu1', bleu1, cur_epoch)
    tb_writer.add_scalar('eval/bleu2', bleu1, cur_epoch)
    tb_writer.add_scalar('eval/bleu3', bleu1, cur_epoch)
    tb_writer.add_scalar('eval/bleu4', bleu1, cur_epoch)
    tb_writer.add_scalar('eval/lv', lv, cur_epoch)
    tb_writer.add_scalar('eval/perfection_prediction', perfect, cur_epoch)
    result = dict()
    result['bleu1'] = bleu1
    result['bleu2'] = bleu2
    result['bleu3'] = bleu3
    result['bleu4'] = bleu4
    result['perfect'] = perfect
    result['lv'] = lv

    logger.info('test valid official: '
                'bleu1 = %.2f | bleu2 = %.2f | bleu3 = %.2f | bleu4 = %.2f |' %
                (bleu1, bleu2, bleu3,bleu4) +
                'perfect = %.2f | lv = %.2f | lengths = %.2f | '  %
                (perfect, lv, len(hypotheses)) +
                'test time = %.2f (s)' % eval_time.time())


    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1

def eval_official_scores(hypotheses, references, copy_info, sources=None,
                    filename=None):
    lengths=[len(i) for i in hypotheses.values()]
    hyps=[i for i in hypotheses.values()]
    refs=[i for i in references.values()]
    bleu1 = list()
    bleu2 = list()
    bleu3 = list()
    bleu4 = list()
    lev = list()
    perfect_num=0
    for i in range(len(hyps)):
        hyp=hyps[i]
        ref=refs[i]
        leng=lengths[i]
        is_perfect=True
        if len(hyp)!=len(ref):
            is_perfect=False
        else:
            for p,q in zip(hyp,ref):
                if p != q:
                    is_perfect=False
                    break
        b1, b2, b3, b4, lv = metrics.evaluate_metrics(ref,hyp,is_perfect)

        if is_perfect:
            perfect_num=perfect_num+1

        bleu1.append(b1)
        if b2 is not None:
            bleu2.append(b2)
        if b3 is not None:
            bleu3.append(b3)
        if b4 is not None:
            bleu4.append(b4)
        lev.append(lv)

    return sum(bleu1)/len(bleu1),sum(bleu2)/len(bleu2),sum(bleu3)/len(bleu3),sum(bleu4)/len(bleu4),sum(lev)/len(lev),perfect_num/len(hyps)


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    tb_writer = SummaryWriter(log_dir='tensorboard/' + args.model_name)
    if args.debug:
        args.test_dataset = args.test_dataset + ".debug"
        for index,i in enumerate(args.train_dataset):
            args.train_dataset[index]=i+".debug"
        args.pretrain_dataset = args.pretrain_dataset + ".debug"
    logger.info(f'fuse type: {args.fuse_type}')
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    if not args.only_test:
        train_exs = []
        if args.pretrain_stage:
            for i in args.pretrain_dataset:
                train_exs.extend(pickle.load(open(i, mode="rb")))
        else:
            for i in args.train_dataset:
                train_exs.extend(pickle.load(open(i, mode="rb")))
        logger.info('Num train examples = %d' % len(train_exs))
    else:
        train_exs=None


    test_exs = pickle.load(open(args.test_dataset,mode="rb"))

    logger.info('Num dev examples = %d' % len(test_exs))

    if args.use_bpe:
        vocabs = util.making_dicts_BPE([train_exs, test_exs], args)
    else:
        vocabs=util.making_dicts([train_exs,test_exs],args)

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = CCSGModel.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = CCSGModel.load(args.model_file,args)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = CCSGModel.load_checkpoint(checkpoint_file, args.cuda, original_args=args)
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = CCSGModel.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, vocabs)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.use_cuda:
        model.cuda()

        if args.parallel:
            model.parallelize()
    else:
        model.cpu()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    if not args.only_test:
        if args.pretrain_stage:
            train_dataset = data.PretrainDataset(train_exs, args, vocabs)
        else:
            train_dataset = data.InCompleteCodeDataset(train_exs, args,vocabs)
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

    if args.micro_change:
        dev_dataset = data.InCompleteCodeDataset(test_exs, args,vocabs,is_eval=False,is_micro=True)
    else:
        dev_dataset = data.InCompleteCodeDataset(test_exs, args, vocabs, is_eval=True)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, dev_loader, model, stats, mode='test', tb_writer=tb_writer)

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if not args.STL_learning:
            if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
                logger.info("Use warmup lrate for the %d epoch, from 0 up to %s." %
                            (args.warmup_epochs, args.learning_rate))
                num_batches = len(train_loader.dataset) // args.batch_size
                warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
                stats['warmup_factor'] = warmup_factor
        else:
            num_batches = len(train_loader.dataset) // args.batch_size
            T_all=num_batches * args.STLR_epochs
            stats['T_all']=T_all


        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            if not args.STL_learning:
                if args.optimizer in ['sgd', 'adam'] and epoch > args.stable_epochs:
                    model.optimizer.param_groups[0]['lr'] = \
                        model.optimizer.param_groups[0]['lr'] * args.lr_decay
                logger.info("lr:{}".format(model.optimizer.param_groups[0]['lr']))

            tb_writer.add_scalar('train/lr', model.optimizer.param_groups[0]['lr'], epoch)
            train(args, train_loader, model, stats, tb_writer=tb_writer)
            if (not args.pretrain_stage) and epoch > args.eval_epochs:
            # if True:
                result = validate_official(args, dev_loader, model, stats, tb_writer=tb_writer)
                # Save best valid
                if result[args.valid_metric] > stats['best_valid']:
                    logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                                (args.valid_metric, result[args.valid_metric],
                                 stats['epoch'], model.updates))
                    model.save(args.model_file)
                    stats['best_valid'] = result[args.valid_metric]
                    stats['no_improvement'] = 0
                else:
                    stats['no_improvement'] += 1
                    if stats['no_improvement'] >= args.early_stop:
                        break
            else:
                model.save(args.model_file)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code of CCSG',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    if args.use_cuda:
        args.cuda = torch.cuda.is_available()
        args.parallel = torch.cuda.device_count() > 1
    else:
        args.cuda = False
        args.parallel = False

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
