""" The kernel components of CCSG"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable

from component.encoders.ggnn import GGNN
from component.modules.embeddings import Embeddings
from component.encoders.transformer import TransformerEncoder
from component.inputters import constants
from component.utils.misc import sequence_mask
from component.modules.modeling_t5 import T5Stack,T5Config



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
        self.sep_embeddings = Embeddings(args.emsize,
                                          args.sep_vocab_size,
                                          constants.PAD)
        self.identi_embeddings = Embeddings(args.emsize,
                                          args.identi_vocab_size,
                                          constants.PAD)



        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)

        self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                               args.emsize)

        self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
                                               args.emsize)

        self.identi_pos_embeddings = nn.Embedding(args.max_tgt_len + 2,
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
            word_rep = self.identi_embeddings(sequence.unsqueeze(2))

            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.identi_pos_embeddings(pos_enc)
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

        config=T5Config(
            d_model=512,
            d_kv=args.d_k,
            d_ff=args.d_ff,
            num_layers=args.nlayers,
            num_decoder_layers=None,
            num_heads=args.num_head,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=False,
        )
        config.is_decoder=False
        config.output_hidden_states=True
        config.output_attentions=True

        self.transformer = T5Stack(config)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                src_mask):
        output_dict = self.transformer(inputs_embeds=input,attention_mask=src_mask)
        layer_outputs=output_dict.hidden_states
        memory_bank = output_dict.last_hidden_state
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

        self.use_seq = args.use_seq
        self.use_gnn = False
        config = T5Config(
            d_model=512,
            d_kv=args.d_k,
            d_ff=args.d_ff,
            num_layers=args.nlayers,
            num_decoder_layers=None,
            num_heads=args.num_head,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=False,
        )
        config.is_decoder = True
        config.output_hidden_states = True
        config.output_attentions = True

        self.transformer = T5Stack(config)

    def count_parameters(self):
       return self.transformer.count_parameters()

    # def init_decoder(self,
    #                  src_lens,
    #                  max_src_len):
    #
    #     return self.transformer.init_state(src_lens, max_src_len, None,None)

    def decode(self,
               memory_bank,  # enc_outputs
               tgt_mask,  # tgt_pad_mask
               tgt_emb,
               src_mask,
               step=None,
               layer_wise_coverage=None):

        s = src_mask.unsqueeze(1).repeat(1,tgt_mask.shape[1],1)
        t = tgt_mask.unsqueeze(2).repeat(1,1,src_mask.shape[1])
        cross_mask=s & t

        output_dict=self.transformer(inputs_embeds=tgt_emb,encoder_hidden_states=memory_bank,
                                     attention_mask=tgt_mask,encoder_attention_mask=cross_mask)


        return output_dict.last_hidden_state, output_dict.cross_attentions

    def forward(self,
                memory_bank,  # enc_outputs
                tgt_pad_mask,  # tgt_pad_mask
                tgt_emb,src_mask):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]

        # state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(memory_bank,  # enc_outputs
                tgt_pad_mask,  # tgt_pad_mask
                tgt_emb,src_mask)

class IdentiDecoder(nn.Module):
    """ The module of Decoder"""

    def __init__(self, args, input_size):
        super(IdentiDecoder, self).__init__()

        self.input_size = input_size

        self.use_seq = args.use_seq
        self.use_gnn = False
        config = T5Config(
            d_model=512,
            d_kv=args.d_k,
            d_ff=args.d_ff,
            num_layers=args.nlayers,
            num_decoder_layers=None,
            num_heads=args.num_head,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=False
        )
        config.is_decoder = True
        config.output_hidden_states = True
        config.output_attentions = True
        config.use_gnn=True

        self.transformer = T5Stack(config)

    def count_parameters(self):
       return self.transformer.count_parameters()

    # def init_decoder(self,
    #                  src_lens,
    #                  max_src_len):
    #
    #     return self.transformer.init_state(src_lens, max_src_len, None,None)

    def decode(self,
               tgt_emb, memory_bank, gnn,
               tgt_mask,
               src_mask, node_mask):

        s = src_mask.unsqueeze(1).repeat(1,tgt_mask.shape[1],1)
        t = tgt_mask.unsqueeze(2).repeat(1,1,src_mask.shape[1])
        cross_mask=s & t

        n = node_mask.unsqueeze(1).repeat(1,tgt_mask.shape[1],1)
        t = tgt_mask.unsqueeze(2).repeat(1,1,node_mask.shape[1])
        node_cross_mask=n & t



        output_dict=self.transformer(inputs_embeds=tgt_emb,encoder_hidden_states=memory_bank,
                                     attention_mask=tgt_mask,encoder_attention_mask=cross_mask,
                                     gnn_hidden_states=gnn,
                                     gnn_attention_mask=node_cross_mask
                                     )


        return output_dict.last_hidden_state, output_dict.cross_attentions

    def forward(self,
                tgt_emb, memory_bank, gnn,
                tgt_mask,
                src_mask, node_mask):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]

        # state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_emb, memory_bank, gnn,
                tgt_mask,
                src_mask, node_mask)




class Network_Template_T5(nn.Module):
    """ The modified Transformer serves as the backbone of the CCSG """

    def __init__(self, args, vocabs):
        """Constructor of the class."""
        super(Network_Template_T5, self).__init__()

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
            'n_steps': 5,
        }
        # create graph encoder, sequence encoder(encoder), decoder,respectively
        self.gnn = GGNN(gnn_args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.decoder = Decoder(args, self.embedder.dec_input_size)
        self.identi_decoder=IdentiDecoder(args, self.embedder.dec_input_size)


        self.layer_wise_attn = args.layer_wise_attn

        self.temp_generator = nn.Linear(self.decoder.input_size, args.sep_vocab_size)

        self.generator = nn.Linear(self.decoder.input_size, args.identi_vocab_size)

        # if args.share_decoder_embeddings:
        #     self.generator.weight = self.embedder.word_embeddings.word_lut.weight


        self.temp_criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # if args.MTL:
        #     self.tokenQuantityRegressor = TokenQuantityRegressor(args)
        #     self.autoWeightedLoss = AutoWeightedLoss(2, args)
        self.args=args
        self.M_token=nn.Linear(args.emsize, args.emsize,bias=False)
        self.M_type = nn.Linear(args.emsize, args.emsize,bias=False)

    def eval_get_template(self,
                        data_tuple):

        (data, (source_map, blank, fill)) = data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments) = data

        src_rep = self.embedder(vec_src,
                                mode='encoder')

        src_mask = sequence_mask(lengths_src, vec_src.shape[1])
        memory_bank, layer_wise_outputs = self.encoder(src_rep, src_mask)  # B x seq_len x h

        enc_outputs = memory_bank

        dec_preds,  _ = self.__generate_template(memory_bank, src_mask, vec_temp.shape[1])
        stack_template = torch.stack(dec_preds, dim=1)
        return enc_outputs,stack_template

    def eval_get_res(self,data_tuples,enc_outputs):
        (data, (source_map, blank, fill)) =data_tuples
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


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

        lengths_src=lengths_src.to(lengths_node.device)
        src_mask = sequence_mask(lengths_src, vec_src.shape[1])
        node_mask = sequence_mask(lengths_node, max_len=vec_attrs.size(1))

        id_dec_preds, _= \
            self.__generate_identi(enc_outputs,gnn,src_mask,vec_tgt.shape[1],node_mask)
        stack_identi = torch.stack(id_dec_preds, dim=1)
        return stack_identi



    def _run_forward_ml(self,
                        data_tuple,mode):


        (data, (source_map, blank, fill)) =data_tuple
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK, vec_temp), edge_metrix, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node, lengths_temp
        ), (src_vocabs, src_map, alignments)=data


        src_rep = self.embedder(vec_src,
                                 mode='encoder')

        batch_size=len(vec_MASK)

        src_mask = sequence_mask(lengths_src, vec_src.shape[1])
        memory_bank, layer_wise_outputs = self.encoder(src_rep, src_mask)  # B x seq_len x h

        enc_outputs =  memory_bank

        if mode=="test":
            dec_preds, _= self.__generate_template(memory_bank,src_mask,vec_temp.shape[1])
            stack_template=torch.stack(dec_preds,dim=1)


        elif mode=="train":
            # todo input gnnencoder

            temp_emb = self.embedder(vec_temp,
                                        mode='sep')
            temp_mask = sequence_mask(lengths_temp, max_len=temp_emb.size(1))


            dec_out, attns = self.decoder(enc_outputs,
                                                     temp_mask,
                                                     temp_emb,src_mask)


            decoder_outputs = dec_out

            loss = dict()

            temp_target = vec_temp[:, 1:].contiguous()

            scores = self.temp_generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            temp_loss = self.temp_criterion(scores.view(-1, scores.size(2)),
                                     temp_target.view(-1))

            temp_loss = temp_loss.view(*scores.size()[:-1])


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

        node_mask = sequence_mask(lengths_node, max_len=vec_attrs.size(1))

        if mode == "test":
            id_dec_preds, id_dec_log_probs=\
                self.__generate_identi(memory_bank,gnn,src_mask,vec_tgt.shape[1],node_mask)
            stack_identi = torch.stack(id_dec_preds, dim=1)
            return {
                "stack_template":stack_template,
                "stack_identi":stack_identi
            }


        # todo input identi decoder

        tgt_emb = self.embedder(vec_tgt,
                                    mode='identi')
        tgt_mask = sequence_mask(lengths_tgt, max_len=tgt_emb.size(1))


        identi_dec_out, _ = self.identi_decoder(tgt_emb,enc_outputs,gnn,
                                                     tgt_mask,
                                                     src_mask,node_mask)

        idt_target = vec_tgt[:, 1:].contiguous()


        identi_scores = self.generator(identi_dec_out)

        identi_scores = identi_scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`


        identi_loss = self.criterion(identi_scores.view(-1, identi_scores.size(2)),
                                 idt_target.view(-1))

        identi_loss = identi_loss.view(*identi_scores.size()[:-1])

        combine_loss=torch.cat((temp_loss * self.args.temp_rate,identi_loss * (1-self.args.temp_rate)),dim=1)


        loss['ml_loss'] = combine_loss  # 32,get average
        # coefs_final = [1, -1]
        # loss['coefs_final'] = coefs_final
        return loss, attns

    def forward(self,data_tuple,mode,enc_outputs=None):
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
            return self.eval_get_res(data_tuple,enc_outputs)
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

    def __generate_identi(self,memory_bank,gnn,src_mask,max_tgt_len,node_mask
                            ):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda


        tgt_words = torch.LongTensor([constants.BOS])
        if use_cuda:
            tgt_words = tgt_words.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1


        dec_preds = []
        dec_log_probs = []


        # +1 for <EOS> token
        for idx in range(max_tgt_len + 1):
            tgt = self.embedder(tgt_words,
                                mode='identi',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(constants.PAD)
            dec_out, attns = self.identi_decoder.decode(tgt, memory_bank, gnn,
                                                                   ~tgt_pad_mask,
                                                                   src_mask, node_mask)
            decoder_outputs = dec_out

            prediction = self.generator(decoder_outputs)
            prediction = f.softmax(prediction[:,-1,:], dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)


            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            tgt_words = torch.cat((tgt_words,tgt),dim=1)

        return dec_preds, dec_log_probs

    def __generate_template(self,
                            memory_bank,src_mask,max_tgt_len):

        batch_size = memory_bank.size(0)
        use_cuda = memory_bank.is_cuda

        dec_preds = []
        dec_log_probs=[]

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
            dec_out, attns = self.decoder.decode(memory_bank,  # enc_outputs
                                                               ~tgt_pad_mask,  # tgt_pad_mask
                                                               tgt,
                                                               src_mask)
            decoder_outputs = dec_out

            prediction = self.temp_generator(decoder_outputs)
            prediction = f.softmax(prediction[:,-1,:], dim=1)


            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)


            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            # if "std" in attns:
            #     # std_attn: batch_size x num_heads x 1 x src_len
            #     std_attn = torch.stack(attns["std"], dim=1)
            #     attentions.append(std_attn.squeeze(2))

            # words = self.__tens2sent(tgt, params['tgt_dict'])
            #
            # words = [params['tgt_dict'][w] for w in words]
            # words = tgt
            tgt_words = torch.cat((tgt_words,tgt),dim=1)

        return dec_preds, dec_log_probs

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
