"""CCSG model"""
import copy
import math
import logging
from tqdm import tqdm

import torch
import torch.optim as optim

import component.config
from torch.nn.utils import clip_grad_norm_

import config
from component.config import override_model_args
from component.models.transformer import Transformer
from component.models.network_template import Network_Template
from component.models.network_OPT import Network_OPT
from component.utils.copy_utils import collapse_copy_scores, \
    make_src_map, align
from component.utils.misc import tens2sen, count_file_lines,convert_temp_and_identi,get_temp_and_identi,replace_unknown,get_temp_and_identi_bpe
from component.utils.misc import get_its_seq
from pretreat.pretreatment import get_graph_ex_from_code
from component.inputters.utils import get_node_representation
from component.inputters import batchify_only_graph
import javalang

logger = logging.getLogger(__name__)


class CCSGModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, vocabs,state_dict=None):
        # Book-keeping.
        self.args = args
        self.args.src_vocab_size = 0
        (self.vocab_attr, self.vocab_type, self.vocab_token,self.vocab_sep,self.vocab_identi)=vocabs
        self.args.type_vocab_size=len(self.vocab_type)
        self.args.vocab_size = self.vocab_token.get_vocab_size()
        if state_dict:
            self.args.vocab_size = state_dict['embedder.word_embeddings.make_embedding.emb_luts.0.weight'].shape[0]

        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        if args.model_type == "transformer":
            self.network = Network_Template(args, vocabs)
        else:
            self.network = Network_OPT(args, vocabs)


        # Load saved state
        if state_dict is not None:
            # Load buffer separately
            self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            self.network.embedder.src_word_embeddings.fix_word_lut()
            self.network.embedder.tgt_word_embeddings.fix_word_lut()

        if self.args.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)

        elif self.args.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def convert_cuda(self,data):
        if isinstance(data,tuple):
            lst=[]
            for i in data:
                lst.append(self.convert_cuda(i))
            return tuple(lst)
        if isinstance(data,list):
            for idx,i in enumerate(data):
                data[idx]=self.convert_cuda(i)
            return data
        elif isinstance(data,dict):
            for key,v in data.items():
                data[key]=self.convert_cuda(v)
            return data
        elif isinstance(data,torch.Tensor):
            return data.cuda(non_blocking=True)
        else:
            return data

    def update(self, ex, epoch):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()
        bsz=ex['batch_size']

        data=ex['data']
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp,vec_mark_range), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        # if self.args.copy_attn:
        #     source_map = make_src_map(src_map)
        #
        #     alignment = align(alignments)
        #     assert (alignments==alignment).all()
        #
        #     blank, fill = collapse_copy_scores(self.vocab_identi, src_vocabs)

        data_tuples=(data,(source_map,blank, fill))
        if self.args.use_cuda and not self.args.parallel:
            data_tuples=self.convert_cuda(data_tuples)



            # Run forward
        if not  self.args.debug:
            try:
                net_loss = self.network(data_tuples,mode="train")
            except Exception as e:
                print(e)
                print("error")

                print("srclen{},tgtlen{},node_dim{}".format(vec_src.shape[1],vec_tgt.shape[1],vec_token.shape[2]))
                return {
                    'ml_loss': -1,
                    'perplexity': -1,
                }
        else:
            net_loss = self.network(data_tuples,mode="train")

            # GNN point
            # loss = net_loss['ml_loss'].mean() if self.parallel \
            #     else net_loss['ml_loss']
            # loss_per_token = net_loss['loss_per_token'].mean() if self.parallel \
            #     else net_loss['loss_per_token']

        loss = net_loss['ml_loss'].mean()
        # loss = net_loss['ml_loss']

        if loss.device.type!="cpu":
            ml_loss = loss.item()
        else:
            ml_loss=loss


        try:
            loss.backward()
        except Exception as e:
            print("srclen{},tgtlen{},node_dim{}".format(vec_src.shape[1], vec_tgt.shape[1], vec_token.shape[2]))
            print("out of memory")

        clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'ml_loss': ml_loss,
            'perplexity': 0,
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------
    def eval(self,ex):
        self.network.eval()
        bsz=ex['batch_size']
        raw_text=ex['raw_text']
        raw_tgt=ex['raw_tgt']
        raw_template=ex["raw_template"]
        data=ex['data']
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp,vec_mark_range), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:

            source_map = make_src_map(src_map)

            alignment = align(alignments)
            assert (alignments==alignment).all()

            blank, fill = collapse_copy_scores(self.vocab_identi, src_vocabs)

        data_tuples=(data,(source_map,blank, fill))

        if self.args.use_cuda and not self.args.parallel:
            data_tuples=self.convert_cuda(data_tuples)

        enc_outputs,stack_template,decoder_out=self.network(data_tuples,mode="eval_temp")
        tp = get_its_seq(stack_template, self.vocab_sep)
        target_MASK_idx = []
        batch_data = []
        for idx,lis in enumerate(tp):
            lengths_temp[idx]=len(lis)
        no_graph_number=0
        for number, item in enumerate(raw_text):
            for idx, tk in enumerate(item):
                if tk == "MASK":
                    target_MASK_idx.append(idx)
                    item.pop(idx)
                    now_insert_idx = idx
                    for i in tp[number]:
                        item.insert(now_insert_idx, i)
                        now_insert_idx = now_insert_idx + 1
                    res = get_graph_ex_from_code(item)
                    if res is not None:
                        node_tokens, node_types, node_attr, edge_dicts = get_node_representation(res, self.vocab_token,
                                                                                             self.vocab_type,
                                                                                             self.vocab_attr)
                    else:
                        node_tokens, node_types, node_attr, edge_dicts = (
                            torch.tensor([[0]], dtype=torch.int64), torch.tensor([0], dtype=torch.int64),
                            torch.tensor([0], dtype=torch.int64),
                            {})
                        no_graph_number=no_graph_number+1
                    batch_b = {
                        "types": node_types,
                        "tokens": node_tokens,
                        "attrs": node_attr,
                        "edge_dicts": edge_dicts
                    }
                    batch_data.append(batch_b)
                    break
        vec_type, vec_token, vec_attrs, edge_metrix, lengths_type, lengths_token, lengths_node = \
            batchify_only_graph(batch_data)
        if self.args.use_cuda and not self.args.parallel:
            (vec_src,vec_type, vec_token, vec_attrs, edge_metrix, lengths_type, lengths_token, lengths_node) = \
                self.convert_cuda(
                    (vec_src,vec_type, vec_token, vec_attrs, edge_metrix, lengths_type, lengths_token, lengths_node))

        data=(vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp,vec_mark_range), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)
        data_tuples=(data, (source_map, blank, fill))

        stack_identi,attentions = self.network(data_tuples, mode="eval_res",enc_outputs=enc_outputs,decoder_out=decoder_out)

        prediction_dict={
            "stack_template":stack_template,
            "stack_identi":stack_identi
        }


        if not self.args.use_bpe:
            template_prediction,identi_prediction,text = get_temp_and_identi(prediction_dict,self.vocab_sep,self.vocab_identi,src_vocabs)
        else:
            template_prediction,identi_prediction,text = get_temp_and_identi_bpe(prediction_dict,self.vocab_sep,self.vocab_token,src_vocabs)

        for i in range(len(identi_prediction)):
            enc_dec_attn = attentions[i]
            if self.args.model_type == 'transformer':
                assert enc_dec_attn.dim() == 3
                enc_dec_attn = enc_dec_attn.mean(1)
            identi_prediction[i] = replace_unknown(identi_prediction[i],
                                             enc_dec_attn,
                                             src_raw=raw_text[i])

        targets = raw_tgt
        return template_prediction,identi_prediction,text,targets,raw_template,no_graph_number




    def predict(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""

        self.network.eval()
        bsz=ex['batch_size']
        raw_text=ex['raw_text']
        raw_tgt=ex['raw_tgt']
        raw_template = ex["raw_template"]
        data=ex['data']
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp,vec_mark_range), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:

            source_map = make_src_map(src_map)

            alignment = align(alignments)
            assert (alignments==alignment).all()

            blank, fill = collapse_copy_scores(self.vocab_identi, src_vocabs)

        data_tuples=(data,(source_map,blank, fill))
        if self.args.use_cuda and not self.args.parallel:
            data_tuples=self.convert_cuda(data_tuples)


        # Run forward
        prediction_dict  = self.network(data_tuples,mode="test")
        identi_prediction = get_temp_and_identi_bpe(prediction_dict,self.vocab_sep,self.vocab_token,src_vocabs)

        # if replace_unk:
        #     for i in range(len(predictions)):
        #         enc_dec_attn = decoder_out['attentions'][i]
        #         assert enc_dec_attn.dim() == 3
        #         enc_dec_attn = enc_dec_attn.mean(1)
        #         predictions[i] = replace_unknown(predictions[i],
        #                                          enc_dec_attn,
        #                                          src_raw=raw_text[i])




        targets = [tarCode for tarCode in raw_tgt]
        return None,identi_prediction,identi_prediction,targets,raw_template


    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        """Save the current checkpoint
        """
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'vocab_attr': self.vocab_attr,
            'vocab_token': self.vocab_token,
            'vocab_type': self.vocab_type,
            'vocab_identi': self.vocab_identi,
            'vocab_sep': self.vocab_sep,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'vocab_type': self.vocab_type,
            'vocab_token': self.vocab_token,
            'vocab_attr': self.vocab_attr,
            'vocab_identi': self.vocab_identi,
            'vocab_sep': self.vocab_sep,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        vocab_attr = saved_params['vocab_attr']
        vocab_token = saved_params['vocab_token']
        vocab_type = saved_params['vocab_type']
        vocab_identi = saved_params['vocab_identi']
        vocab_sep = saved_params['vocab_sep']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        vocabs=(vocab_attr, vocab_type, vocab_token,vocab_sep,vocab_identi)
        return CCSGModel(args, vocabs,state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True, original_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )

        vocab_type=saved_params['vocab_type']
        vocab_token = saved_params['vocab_token']
        vocab_attr = saved_params['vocab_attr']
        vocab_sep=saved_params['vocab_sep']
        vocab_identi=saved_params['vocab_identi']


        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        args = saved_params['args']



        args_attr=set.union(config.MODEL_ARCHITECTURE,config.DATA_OPTIONS,config.MODEL_OPTIMIZER)
        for attr in args_attr:
            if not hasattr(args,attr):
                value=getattr(original_args,attr,False)
                setattr(args,attr,value)

        args.use_cuda=original_args.use_cuda
        args.parallel=original_args.parallel
        args.debug = original_args.debug
        if args.learning_rate != original_args.learning_rate:
            optimizer['param_groups'][0]['lr'] = original_args.learning_rate
            print("learning rate reset")
        args.fine_tune_template=original_args.fine_tune_template

        vocabs=(vocab_attr, vocab_type, vocab_token,vocab_sep,vocab_identi)
        model = CCSGModel(args, vocabs,state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
