""" The kernel components of CCSG"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable

from component.encoders.ggnn import GGNN
from component.modules.embeddings import Embeddings
from component.encoders.transformer import TransformerEncoder
from component.decoders.transformer import TransformerDecoder
from component.inputters import constants
from component.modules.global_attention import GlobalAttention
from component.modules.copy_generator import CopyGenerator, CopyGeneratorCriterion
from component.utils.misc import sequence_mask


class Embedder(nn.Module):
    """ An Embedder that provides embedding for all inputs"""

    def __init__(self, args):
        super(Embedder, self).__init__()
        self.enc_input_size=args.emsize
        self.dec_input_size=args.emsize

        self.word_embeddings = Embeddings(args.emsize,
                                              args.vocab_size,
                                              constants.PAD)
        self.type_embeddings = Embeddings(args.emsize,
                                              args.type_vocab_size,
                                              constants.PAD)
        self.attr_embeddings = Embeddings(args.emsize,
                                              args.attr_vocab_size,
                                              constants.PAD)


        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)

        self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                               args.emsize)

        self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                               args.emsize)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                mode='encoder',step=None):

        if mode == 'encoder':
            word_rep = self.word_embeddings(sequence.unsqueeze(2))
            if self.src_pos_emb and self.no_relative_pos:
                pos_enc = torch.arange(start=0,
                                       end=word_rep.size(1)).type(torch.LongTensor)
                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.src_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'decoder':
            word_rep = self.word_embeddings(sequence.unsqueeze(2))

            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.tgt_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'token':
            word_rep = self.word_embeddings(sequence.unsqueeze(2))
        elif mode == 'type':
            word_rep = self.type_embeddings(sequence.unsqueeze(2))
        elif mode == 'attrs':
            word_rep=self.attr_embeddings(sequence.unsqueeze(2))

        else:
            raise ValueError('Unknown embedder mode!')

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    """"The module of Sequence Encoder"""

    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(num_layers=args.nlayers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                input_len):
        layer_outputs, _ = self.transformer(input, input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class GNNEncoder(nn.Module):
    """The module of Graph Encoder"""

    def __init__(self,
                 args,
                 nlayers,
                 input_size):
        super(GNNEncoder, self).__init__()

        max_relative_pos = [args.max_relative_pos[0]] * nlayers
        self.transformer = TransformerEncoder(num_layers=nlayers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                input_len):
        layer_outputs, _ = self.transformer(input, input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Decoder(nn.Module):
    """ The module of Decoder"""

    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        self.use_seq = args.use_seq
        self.use_gnn = args.use_gnn
        if self.split_decoder:
            # Following (https://arxiv.org/pdf/1808.07913.pdf), we try to split decoder
            self.transformer_c = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )
            self.transformer_d = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                dropout=args.trans_drop
            )

            # To accomplish eq. 19 - 21 from `https://arxiv.org/pdf/1808.07913.pdf`
            self.fusion_sigmoid = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.ReLU()
            )
        else:
            self.transformer = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )

        if args.reload_decoder_state:
            state_dict = torch.load(
                args.reload_decoder_state, map_location=lambda storage, loc: storage
            )
            self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        if self.split_decoder:
            return self.transformer_c.count_parameters() + self.transformer_d.count_parameters()
        else:
            return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len, lengths_node=None, max_node_len=None):

        if self.split_decoder:
            state_c = self.transformer_c.init_state(src_lens, max_src_len)
            state_d = self.transformer_d.init_state(src_lens, max_src_len)
            return state_c, state_d
        else:
            return self.transformer.init_state(src_lens, max_src_len, lengths_node, max_node_len)

    def decode(self,
               tgt_words,  # tgt_mask
               tgt_emb,
               memory_bank,  # encoder_output
               state,
               gnn=None,
               template_out=None,
               step=None,
               layer_wise_coverage=None):


        decoder_outputs, attns = self.transformer(tgt_words,
                                                      tgt_emb,
                                                      memory_bank if self.use_seq else None,
                                                      state,
                                                      gnn=gnn if self.use_gnn else None,
                                                      template_out=template_out,
                                                      step=step,
                                                      layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,  # enc_outputs
                memory_len,  # code_len
                tgt_pad_mask,  # tgt_pad_mask
                tgt_emb,  # tgt_emb
                gnn, lengths_node,template_out=None):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        if gnn is not None:
            max_node_len = gnn[0].shape[1] \
                if isinstance(gnn, list) else gnn.shape[1]
        else:
            max_node_len = None
        state = self.init_decoder(memory_len, max_mem_len, lengths_node, max_node_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state, gnn=gnn,template_out=template_out)


class Transformer(nn.Module):
    """ The modified Transformer serves as the backbone of the CCSG """

    def __init__(self, args, tgt_dict):
        """Constructor of the class."""
        super(Transformer, self).__init__()

        self.name = 'Transformer'

        # create embedders
        self.embedder = Embedder(args)
        # self.MTL = args.MTL

        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers

        gnn_args = {
            'state_dim': args.emsize,
            'n_edge_types': len(constants.edge_type),
            'n_steps': 5,
        }
        # create graph encoder, sequence encoder(encoder), decoder,respectively
        self.gnn = GGNN(gnn_args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.decoder = Decoder(args, self.embedder.dec_input_size)
        self.layer_wise_attn = args.layer_wise_attn

        self.generator = nn.Linear(self.decoder.input_size, args.vocab_size)
        if args.share_decoder_embeddings:
            self.generator.weight = self.embedder.word_embeddings.word_lut.weight

        self._copy = args.copy_attn
        if self._copy:
            self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
                                             attn_type=args.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                tgt_dict,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=len(tgt_dict),
                                                    force_copy=args.force_copy)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        # if args.MTL:
        #     self.tokenQuantityRegressor = TokenQuantityRegressor(args)
        #     self.autoWeightedLoss = AutoWeightedLoss(2, args)
        self.args=args
        self.M_token=nn.Linear(args.emsize, args.emsize,bias=False)
        self.M_type = nn.Linear(args.emsize, args.emsize,bias=False)
        self.token_vocab=tgt_dict

    def _run_forward_ml(self,
                        data_tuple,mode):


        (data, (source_map, blank, fill)) =data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node
        ), (src_vocabs, src_map, alignments)=data


        src_rep = self.embedder(vec_src,
                                 mode='encoder')

        batch_size=len(vec_MASK)

        memory_bank, layer_wise_outputs = self.encoder(src_rep, lengths_src)  # B x seq_len x h

        # if self.MTL:
        #     tarCode_len_labels = tarCode_len.to(torch.float32)
        #     quantity_out, mse_loss = self.tokenQuantityRegressor(memory_bank, tarCode_len_labels)

        vec_full_type=torch.zeros(size=vec_attrs.shape,dtype=torch.long,device=vec_attrs.device)

        vec_full_token=torch.zeros(size=vec_attrs.shape,dtype=torch.long,device=vec_attrs.device)
        for idx in range(len(lengths_token)):
            l_type=lengths_type[idx]
            l_token=lengths_token[idx]
            vec_full_type[idx].scatter_(0, torch.arange(0, l_type, device=vec_type.device, dtype=torch.int64),
                                         vec_type[idx])
            vec_full_token[idx].scatter_(0,torch.arange(l_type,l_type+l_token,device=vec_token.device,dtype=torch.int64),vec_token[idx])

        type_rep=self.embedder(vec_full_type,
                                 mode='type')
        token_rep=self.embedder(vec_full_token,
                                 mode='token')
        attr_rep=self.embedder(vec_attrs,
                                 mode='attrs')

        node_val=self.M_type(type_rep)+self.M_token(token_rep)

        node_val=node_val+attr_rep

        gnn_output = self.gnn(node_val, edge_metrix)

        gnn = gnn_output

        if mode == "test":
            params = dict()
            params['memory_bank'] = memory_bank
            params['layer_wise_outputs'] = layer_wise_outputs
            params['src_len'] = lengths_src
            params['source_vocab'] = src_vocabs
            params['src_map'] = source_map

            params['fill'] = fill
            params['blank'] = blank
            params['src_dict'] = self.token_vocab
            params['tgt_dict'] = self.token_vocab
            params['max_len'] = vec_tgt.shape[1]
            params['src_words'] = vec_src
            params['gnn'] = gnn
            params['node_len'] = lengths_node

            dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
            dec_preds = torch.stack(dec_preds, dim=1)
            copy_info = torch.stack(copy_info, dim=1) if copy_info else None
            # attentions: batch_size x tgt_len x num_heads x src_len
            attentions = torch.stack(attentions, dim=1) if attentions else None

            return {
                'predictions': dec_preds,
                'copy_info': copy_info,
                'memory_bank': memory_bank,
                'attentions': attentions}

        # todo input gnnencoder

        tarCode_emb = self.embedder(vec_tgt,
                                    mode='decoder')
        tarCode_pad_mask = ~sequence_mask(lengths_tgt, max_len=tarCode_emb.size(1))


        enc_outputs = layer_wise_outputs if self.layer_wise_attn else memory_bank


        layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                 lengths_src,
                                                 tarCode_pad_mask,
                                                 tarCode_emb, gnn, lengths_node)

        decoder_outputs = layer_wise_dec_out[-1]

        loss = dict()

        target = vec_tgt[:, 1:].contiguous()
        if self._copy:

            _, copy_score, _ = self.copy_attn(decoder_outputs,
                                              memory_bank,
                                              memory_lengths=lengths_src,
                                              softmax_weights=False)

            attn_copy = f.softmax(copy_score, dim=-1)
            scores = self.copy_generator(decoder_outputs, attn_copy, source_map)
            scores = scores[:, :-1, :].contiguous()
            ml_loss = self.criterion(scores,
                                     alignments[:, 1:].contiguous(),
                                     target)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(scores.view(-1, scores.size(2)),
                                     target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])

        loss['loss_per_token'] = ml_loss.div((lengths_tgt.unsqueeze(dim=1)).float())
        # coefs_final = [0, 0]
        # if self.MTL:
        #     cache_weight = self.autoWeightedLoss.constant_weight[0] + (self.autoWeightedLoss.epoch - 1) * 0.02
        #     if cache_weight < 1:
        #         coefs_final[0] = ml_loss.mean()
        #         coefs_final[1] = mse_loss
        #         self.autoWeightedLoss.epoch = torch.tensor(epoch).to(mse_loss.device)
        #         final_loss, coefs_final = self.autoWeightedLoss([ml_loss.mean(), mse_loss])
        #         loss['ml_loss'] = final_loss
        #     else:
        #         loss['ml_loss'] = ml_loss.mean()  # 32,get average
        #         coefs_final = [1, -1]
        # else:
        loss['ml_loss'] = ml_loss  # 32,get average
        # coefs_final = [1, -1]
        # loss['coefs_final'] = coefs_final
        return loss, attns

    def forward(self,data_tuple):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - tarCode_word_rep: ``(batch_size, max_que_len)``
            - tarCode_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - tarCode_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(data_tuple,mode="train")

        else:
            return self._run_forward_ml(data_tuple,mode="test")

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([constants.BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        lengths_node = params['node_len']
        gnn = params['gnn']

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]

        max_node_len = params['gnn'][0].shape[1] \
            if isinstance(params['gnn'], list) else params['gnn'].shape[1]
        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len, lengths_node, max_node_len)

        attns = {"coverage": None}
        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']

        # +1 for <EOS> token
        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            gnn=gnn,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self._copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  params['memory_bank'],
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                attn_copy = f.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))
            if self._copy:
                mask = tgt.gt(len(params['tgt_dict']) - 1)
                copy_info.append(mask.float().squeeze(1))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])

            words = [params['tgt_dict'][w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, copy_info, dec_log_probs

    def decode(self,
               code_word_rep,
               code_len,
               src_map,
               alignment,
               **kwargs):

        batch_size = code_len.size(0)
        nodes = kwargs['nodes']  # torch.Size([16, 150]),word_dict[w] for w in self.tokens
        adjacency_matrix = kwargs['adjacency_matrix']  # torch.Size([16, 150, 1200])
        backbone_sequence = kwargs['backbone_sequence']
        max_lengths = kwargs['max_lengths']
        lengths_backbone = kwargs['lengths_backbone']
        lengths_node = kwargs['lengths_node']
        type_sequence = kwargs['type_sequence']
        mask_id_matrix = kwargs['mask_id_matrix']
        virtual_index_matrix = kwargs['virtual_index_matrix']
        mask_relative = kwargs['mask_relative']

        code_rep = self.embedder(code_word_rep,
                                 None,
                                 mode='encoder')
        memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h

        nodes_feature = self.embedder(nodes, None, mode='gnn',
                                      type_sequence=type_sequence)  # torch.Size([16, 150, 512])
        gnn_output = self.gnn(nodes_feature, adjacency_matrix)  # torch.Size([16, 150, 512])

        gnn = gnn_output

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = layer_wise_outputs
        params['src_len'] = code_len
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map

        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']
        params['src_dict'] = kwargs['src_dict']
        params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = kwargs['max_len']
        params['src_words'] = code_word_rep
        params['gnn'] = gnn
        params['node_len'] = lengths_node

        dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
