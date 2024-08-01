""" The kernel components of CCSG"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable

from component.encoders.ggnn import GGNN
from component.encoders.ggnn_encoder_ver import GGNN_ENC
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


        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)

        self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                               args.emsize)

        self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                               args.emsize)

        # self.identi_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
        #                                        args.emsize)


        self.dropout = nn.Dropout(args.dropout_emb)
        self.args=args

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

        elif mode == 'sep':
            word_rep = self.sep_embeddings(sequence.unsqueeze(2))

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

        elif mode == 'identi':
            if  not self.args.use_bpe:
                word_rep = self.identi_embeddings(sequence.unsqueeze(2))
            else:
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
        # self.transformer = TransformerEncoder(num_layers=12,
        #                                       d_model=768,
        #                                       heads=12,
        #                                       d_k=args.d_k,
        #                                       d_v=args.d_v,
        #                                       d_ff=args.d_ff,
        #                                       dropout=args.trans_drop,
        #                                       max_relative_positions=64,
        #                                       use_neg_dist=args.use_neg_dist)
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
        memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Decoder(nn.Module):
    """ The module of Decoder"""

    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        self.use_seq = args.use_seq
        self.use_gnn = False
        self.transformer = TransformerDecoder(
            num_layers=args.nlayers,
            d_model=self.input_size,
            heads=args.num_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            coverage_attn=args.coverage_attn,
            dropout=args.trans_drop,
            has_gnn_attn=False
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
                     max_src_len):

        return self.transformer.init_state(src_lens, max_src_len, None,None)

    def decode(self,
               tgt_words,  # tgt_mask
               tgt_emb,
               memory_bank,  # encoder_output
               state,
               step=None,
               layer_wise_coverage=None,
               ):

        decoder_outputs, attns = self.transformer(tgt_words,
                                                  tgt_emb,
                                                  memory_bank ,
                                                  state,
                                                  gnn=None,
                                                  step=step,
                                                  layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,  # enc_outputs
                memory_len,  # code_len
                tgt_pad_mask,  # tgt_pad_mask
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]

        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)


class IdentiDecoder(nn.Module):
    """ The module of Decoder"""

    def __init__(self, args, input_size):
        super(IdentiDecoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        self.use_seq = args.use_seq
        self.use_gnn = args.use_gnn
        self.transformer = TransformerDecoder(
            num_layers=args.nlayers,
            d_model=self.input_size,
            heads=args.num_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            coverage_attn=args.coverage_attn,
            dropout=args.trans_drop,
            has_gnn_attn=True
        )


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
               gnn,
               step=None,
               layer_wise_coverage=None):

        decoder_outputs, attns = self.transformer(tgt_words,
                                                  tgt_emb,
                                                  memory_bank ,
                                                  state,
                                                  gnn,
                                                  step=step,
                                                  layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,  # enc_outputs
                memory_len,  # code_len
                tgt_pad_mask,  # tgt_pad_mask
                tgt_emb,  # tgt_emb
                gnn, lengths_node):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        max_node_len = gnn[0].shape[1] \
            if isinstance(gnn, list) else gnn.shape[1]

        state = self.init_decoder(memory_len, max_mem_len, lengths_node, max_node_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state, gnn=gnn)


class Network_Template(nn.Module):
    """ The modified Transformer serves as the backbone of the CCSG """

    def __init__(self, args, vocabs):
        """Constructor of the class."""
        super(Network_Template, self).__init__()

        self.name = 'Transformer'
        (self.vocab_attr, self.vocab_type, self.vocab_token, self.vocab_sep, self.vocab_identi) = vocabs
        # create embedders
        self.embedder = Embedder(args)
        # self.MTL = args.MTL

        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers

        gnn_args = {
            'state_dim': args.emsize,
            'n_edge_types': len(constants.edge_type),
            'n_steps': 6,
        }
        # create graph encoder, sequence encoder(encoder), decoder,respectively
        self.gnn = GGNN(gnn_args)
        # self.gnn_encoder=GNNEncoder(args,
        #          3,
        #          args.emsize)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.identi_decoder=IdentiDecoder(args, self.embedder.dec_input_size)


        self.layer_wise_attn = args.layer_wise_attn


        self.generator = nn.Linear(self.identi_decoder.input_size, args.vocab_size)

        # if args.share_decoder_embeddings:
        #     self.generator.weight = self.embedder.word_embeddings.word_lut.weight

        self._copy = args.copy_attn
        if self._copy:
            self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
                                             attn_type=args.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                self.vocab_identi,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=len(self.vocab_identi),
                                                    force_copy=args.force_copy)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')


        # if args.MTL:
        #     self.tokenQuantityRegressor = TokenQuantityRegressor(args)
        #     self.autoWeightedLoss = AutoWeightedLoss(2, args)
        self.args=args
        self.M_token=nn.Linear(args.emsize, args.emsize,bias=False)
        self.M_type = nn.Linear(args.emsize, args.emsize,bias=False)
        if not args.no_field_emb:
            self.range_embedding = nn.Embedding(3, args.emsize,padding_idx=0)

    def eval_get_template(self,
                        data_tuple):

        (data, (source_map, blank, fill)) = data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments) = data

        src_rep = self.embedder(vec_src,
                                mode='encoder')


        memory_bank, layer_wise_outputs = self.encoder(src_rep, lengths_src)  # B x seq_len x h

        enc_outputs = memory_bank

        dec_preds, _, _, decoder_out = self.__generate_template(memory_bank, lengths_src, vec_temp.shape[1])
        stack_template = torch.stack(dec_preds, dim=1)
        return enc_outputs,stack_template,decoder_out

    def eval_get_res(self,data_tuples,enc_outputs,decoder_out):
        (data, (source_map, blank, fill)) =data_tuples
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data

        dvs=decoder_out.device

        vec_src_unk=torch.zeros(size=vec_src.shape,dtype=vec_src.dtype,device=dvs)
        vec_src_unk.masked_fill_(vec_src==1,1)
        vec_full_type=torch.zeros(size=vec_attrs.shape,dtype=torch.long,device=vec_attrs.device)

        vec_full_token = torch.zeros(size=[vec_attrs.shape[0], vec_attrs.shape[1], vec_token.shape[2]],
                                     dtype=torch.long, device=vec_attrs.device)
        for idx in range(len(lengths_token)):
            l_type = lengths_type[idx]
            l_token = lengths_token[idx]
            vec_full_type[idx].scatter_(0, torch.arange(0, l_type, device=vec_type.device, dtype=torch.int64),
                                        vec_type[idx])
            # vec_full_token[idx].scatter_(0,torch.arange(l_type,l_type+l_token,device=vec_token.device,dtype=torch.int64),vec_token[idx])
            token_field = torch.arange(l_type, l_type + l_token, device=vec_token.device, dtype=torch.int64).unsqueeze(
                1).repeat(1, vec_token.shape[2])
            vec_full_token[idx].scatter_(0, token_field, vec_token[idx])

        non_0 = vec_full_token.count_nonzero(-1)
        non_0 = torch.where(non_0==0,1,non_0)

        type_rep=self.embedder(vec_full_type,
                                 mode='type')
        token_rep=self.embedder(vec_full_token,
                                 mode='token')

        token_rep = torch.div(token_rep.sum(dim=2), non_0.unsqueeze(2).repeat(1, 1, 512))


        attr_rep=self.embedder(vec_attrs,
                                 mode='attrs')

        node_val=self.M_type(type_rep)+self.M_token(token_rep)

        node_val=node_val+attr_rep

        node_mask = ~sequence_mask(lengths_node, attr_rep.shape[1])

        gnn_output = self.gnn(node_val, edge_metrix,node_mask)

        gnn = gnn_output

        lengths_src=lengths_src.to(dvs)

        tp_out_mask=  ~sequence_mask(lengths_temp, max_len=vec_tgt.size(1)+1)
        tp_out_mask=tp_out_mask.to(dvs)

        id_dec_preds, id_attentions,  id_dec_log_probs=\
            self.__generate_identi(enc_outputs,lengths_node,gnn,lengths_src,vec_tgt.shape[1])
        stack_identi = torch.stack(id_dec_preds, dim=1)
        attentions=torch.stack(id_attentions,dim=1)
        attention_msk=torch.mul(attentions.permute(1,2,0,3),vec_src_unk).permute(2,0,1,3)
        attentions=attention_msk


        return stack_identi,attentions

    def fine_tune_by_template(self,
                        data_tuple):


        (data, (source_map, blank, fill)) =data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        src_rep = self.embedder(vec_src,
                                 mode='encoder')


        # with torch.no_grad():
        memory_bank, layer_wise_outputs = self.encoder(src_rep, lengths_src)  # B x seq_len x h

        enc_outputs =  memory_bank

        temp_emb = self.embedder(vec_temp,
                                    mode='sep')
        temp_mask = ~sequence_mask(lengths_temp, max_len=temp_emb.size(1))

        layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                 lengths_src,
                                                 temp_mask,
                                                 temp_emb)


        decoder_outputs = layer_wise_dec_out[-1]

        temp_target = vec_temp[:, 1:].contiguous()

        scores = self.temp_generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
        scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
        temp_loss = self.temp_criterion(scores.view(-1, scores.size(2)),
                                 temp_target.view(-1))

        temp_loss = temp_loss.view(*scores.size()[:-1])

        return temp_loss



    def _run_forward_ml(self,
                        data_tuple,mode):


        (data, (source_map, blank, fill)) =data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp,vec_mark_range), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        src_rep = self.embedder(vec_src,
                                 mode='encoder')


        enc_outputs, layer_wise_outputs = self.encoder(src_rep, lengths_src)  # B x seq_len x h

        # if mode=="test":
        #     dec_preds, attentions, copy_info, _ = self.__generate_template(memory_bank,lengths_src,vec_temp.shape[1])
        #     stack_template=torch.stack(dec_preds,dim=1)


        # elif mode=="train":
        # todo input gnnencoder
        max_node_len=edge_metrix.shape[1]

        full_shape=(vec_src.shape[0],int(max_node_len))
        vec_full_type=torch.zeros(size=full_shape,dtype=torch.long,device=lengths_node.device)

        vec_full_token = torch.zeros(size=[full_shape[0], full_shape[1], vec_token.shape[2]],
                                     dtype=torch.long, device=lengths_node.device)
        for idx in range(len(lengths_token)):
            l_type=lengths_type[idx]
            l_token=lengths_token[idx]
            vec_full_type[idx].scatter_(0, torch.arange(0, l_type, device=vec_type.device, dtype=torch.int64),
                                         vec_type[idx])
            # vec_full_token[idx].scatter_(0,torch.arange(l_type,l_type+l_token,device=vec_token.device,dtype=torch.int64),vec_token[idx])
            token_field=torch.arange(l_type,l_type+l_token,device=vec_token.device,dtype=torch.int64).unsqueeze(1).repeat(1,vec_token.shape[2])
            vec_full_token[idx].scatter_(0, token_field, vec_token[idx])


        non_0 = vec_full_token.count_nonzero(-1)
        non_0 = torch.where(non_0==0,1,non_0)

        vec_full_type=self.embedder(vec_full_type,
                                 mode='type')
        vec_full_token=self.embedder(vec_full_token,
                                 mode='token')

        vec_full_token = torch.div(vec_full_token.sum(dim=2), non_0.unsqueeze(2).repeat(1, 1, 512))




        node_val=self.M_type(vec_full_type)+self.M_token(vec_full_token)
        if not self.args.no_field_emb:
            vec_mark_range = self.range_embedding(vec_mark_range)
            node_val = node_val+vec_mark_range
        gnn = self.gnn(node_val, edge_metrix)

        if mode == "test":
            id_dec_preds, id_attentions,  id_dec_log_probs=\
                self.__generate_identi(enc_outputs,lengths_node,gnn,lengths_src,vec_tgt.shape[1])
            stack_identi = torch.stack(id_dec_preds, dim=1)
            return {
                "stack_identi":stack_identi
            }


        # todo input identi decoder

        tgt_emb = self.embedder(vec_tgt,
                                    mode='identi')
        tgt_mask = ~sequence_mask(lengths_tgt, max_len=tgt_emb.size(1))

        layer_wise_identi_dec_out, _ = self.identi_decoder(enc_outputs,
                                                 lengths_src,
                                                 tgt_mask,
                                                 tgt_emb, gnn, lengths_node)

        idt_target = vec_tgt[:, 1:].contiguous()


        identi_scores = self.generator(layer_wise_identi_dec_out[-1])

        identi_scores = identi_scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`


        identi_loss = self.criterion(identi_scores.view(-1, identi_scores.size(2)),
                                 idt_target.view(-1))

        identi_loss = identi_loss.view(*identi_scores.size()[:-1])


        combine_loss=identi_loss

        loss={}
        loss['ml_loss'] = combine_loss  # 32,get average
        # coefs_final = [1, -1]
        # loss['coefs_final'] = coefs_final
        return loss

    def forward(self,data_tuple,mode,enc_outputs=None,decoder_out=None):
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

        if mode=="eval_temp":
            return self.eval_get_template(data_tuple)
        elif mode=="eval_res":
            return self.eval_get_res(data_tuple,enc_outputs,decoder_out=decoder_out)
        elif mode=="template":
            return self.fine_tune_by_template(data_tuple)
        else:
            return self._run_forward_ml(data_tuple,mode=mode)

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

    def __generate_identi_new(self,memory_bank,lengths_node,gnn,src_len,max_tgt_len):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda


        tgt_words = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt_words = tgt_words.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        dec_preds = []
        # attentions = []
        dec_log_probs = []


        # max_mem_len = memory_bank.shape[1]
        #
        # max_node_len = gnn.shape[1]
        # # dec_states = self.identi_decoder.init_decoder(src_len, max_mem_len, lengths_node, max_node_len)

        attns = {"coverage": None}
        enc_outputs = memory_bank

        # +1 for <EOS> token
        for idx in range(max_tgt_len + 1):
            tgt = self.embedder(tgt_words,
                                mode='identi',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)

            layer_wise_dec_out, attns = self.identi_decoder(enc_outputs,
                                                 src_len,
                                                 tgt_pad_mask,
                                                 tgt, gnn, lengths_node)

            decoder_outputs = layer_wise_dec_out[-1]


            prediction = self.generator(decoder_outputs)
            prediction = f.softmax(prediction[:,-1,:], dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())


            tgt_words = torch.cat((tgt_words,tgt),dim=1)

        attentions=attns["std"]
        return dec_preds, attentions,  dec_log_probs

    def __generate_template_new(self,memory_bank,src_len,max_tgt_len):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda


        tgt_words = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt_words = tgt_words.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        dec_preds = []
        # attentions = []
        dec_log_probs = []


        # max_mem_len = memory_bank.shape[1]
        #
        # max_node_len = gnn.shape[1]
        # # dec_states = self.identi_decoder.init_decoder(src_len, max_mem_len, lengths_node, max_node_len)

        attns = {"coverage": None}
        enc_outputs = memory_bank

        # +1 for <EOS> token
        for idx in range(max_tgt_len + 1):
            tgt = self.embedder(tgt_words,
                                mode='sep',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)

            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                 src_len,
                                                 tgt_pad_mask,
                                                 tgt)


            decoder_outputs = layer_wise_dec_out[-1]


            prediction = self.temp_generator(decoder_outputs)
            prediction = f.softmax(prediction[:,-1,:], dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())


            tgt_words = torch.cat((tgt_words,tgt),dim=1)

        attentions=attns["std"]
        return dec_preds, attentions,  dec_log_probs,decoder_outputs

    def __generate_identi(self,memory_bank,lengths_node,gnn,src_len,max_tgt_len
                            ):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda


        tgt_words = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt_words = tgt_words.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        dec_preds = []
        attentions = []
        dec_log_probs = []


        max_mem_len = memory_bank.shape[1]

        max_node_len = gnn.shape[1]
        dec_states = self.identi_decoder.init_decoder(src_len, max_mem_len, lengths_node, max_node_len)

        attns = {"coverage": None}
        enc_outputs = memory_bank

        # +1 for <EOS> token
        for idx in range(max_tgt_len + 1):
            tgt = self.embedder(tgt_words,
                                mode='identi',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.identi_decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            gnn=gnn,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]


            prediction = self.generator(decoder_outputs.squeeze(1))
            prediction = f.softmax(prediction, dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            tgt_words = tgt

        return dec_preds, attentions,  dec_log_probs

    def __generate_template(self,
                            memory_bank,src_len,max_tgt_len):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []

        max_mem_len = memory_bank.shape[1]

        dec_states = self.decoder.init_decoder(src_len, max_mem_len)

        attns = {"coverage": None}
        enc_outputs = memory_bank

        tgt_words = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt_words = tgt_words.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        # +1 for <EOS> token
        for idx in range(max_tgt_len + 1):
            tgt = self.embedder(tgt_words,
                                mode='sep',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]

            prediction = self.temp_generator(decoder_outputs.squeeze(1))
            prediction = f.softmax(prediction, dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)


            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            # words = self.__tens2sent(tgt, params['tgt_dict'])
            #
            # words = [params['tgt_dict'][w] for w in words]
            # words = tgt
            tgt_words = tgt

        return dec_preds, attentions,  dec_log_probs, decoder_outputs

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
        return self.identi_decoder.count_parameters()

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
