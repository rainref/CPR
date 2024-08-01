# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/misc.py
# -*- coding: utf-8 -*-

import string
import torch
import subprocess
from nltk.stem import PorterStemmer
from component.inputters import constants

ps = PorterStemmer()


def normalize_string(s, dostem=False):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def stem(text):
        if not dostem:
            return text
        return ' '.join([ps.stem(w) for w in text.split()])

    return stem(white_space_fix(remove_punc(lower(s))))


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def validate(sequence):
    seq_wo_punc = sequence.translate(str.maketrans('', '', string.punctuation))
    return len(seq_wo_punc.strip()) > 0


def tens2sen(t, word_dict=None, src_vocabs=None):
    sentences = []
    # loop over the batch elements
    for idx, s in enumerate(t):
        sentence = []
        for wt in s:
            word = wt if isinstance(wt, int) \
                else wt.item()
            if word in [constants.BOS]:
                continue
            if word in [constants.EOS]:
                break
            if word_dict and word < len(word_dict):
                sentence += [word_dict[word]]
            elif src_vocabs:
                word = word - len(word_dict)
                sentence += [src_vocabs[idx][word]]
            else:
                sentence += [str(word)]

        if len(sentence) == 0:
            # NOTE: just a trick not to score empty sentence
            # this has no consequence
            sentence = [str(constants.PAD)]

        # sentence = ' '.join(sentence)
        # if not validate(sentence):
        #     sentence = str(constants.PAD)
        sentences += [sentence]
    return sentences

def tens2word(t, word_dict=None, src_vocabs=None):
    all_data_result = []
    # loop over the batch elements
    for idx, s in enumerate(t):
        subtokens = []
        for wt in s:
            word = wt if isinstance(wt, int) \
                else wt.item()
            if word in [constants.BOS]:
                continue
            if word in [constants.EOS]:
                break
            if word_dict and word < len(word_dict):
                subtokens += [word_dict[word]]
            elif src_vocabs:
                word = word - len(word_dict)
                subtokens += [src_vocabs[idx][word]]
            else:
                subtokens += [str(word)]

        if len(subtokens) == 0:
            # NOTE: just a trick not to score empty sentence
            # this has no consequence
            sentence = [str(constants.PAD)]

        subtokens = ' '.join(subtokens)
        # if not validate(sentence):
        #     sentence = str(constants.PAD)
        all_data_result += [subtokens]
    return all_data_result


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)  # (0 for pad positions)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
           (hasattr(opt, 'gpu') and opt.gpu > -1)


def generate_relative_positions_matrix(length,
                                       max_relative_positions,
                                       use_neg_dist,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)

    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)

    # Shift values to be >= 0
    if use_neg_dist:
        final_mat = distance_mat_clipped + max_relative_positions
    else:
        final_mat = torch.abs(distance_mat_clipped)

    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def count_gz_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    # cmd = "zcat " + file_path + " | sed -n '$=' "
    cmd = "zcat " + file_path + " | wc -l"
    num = subprocess.check_output(cmd, shell=True)
    # num = subprocess.check_output(['zcat', file_path, '|', 'sed', "-n '$='"])
    num = num.decode('utf-8').split(' ')
    return int(num[0])
def convert_temp_and_identi_new(prediction_dict,vocab_sep,vocab_identi):
    stack_template = prediction_dict['stack_template']
    stack_identi = prediction_dict['stack_identi']
    identi_prediction = []
    for idx, i in enumerate(stack_identi):
        pred = []
        for j in i:
            if j != constants.EOS:
                pred.append(vocab_identi[int(j)])
            else:
                break
        identi_prediction.append(pred)

    text = []
    for idx,i in enumerate(stack_template):
        txt=[]
        number_mask=0

        for jdx,j in enumerate(i):
            if j == constants.EOS:
                break
            txt.append(vocab_sep[int(j)])
        text.append(txt)

def replace_unknown(prediction, attn, src_raw):
    """ ?
        attn: tgt_len x src_len
    """
    tokens = prediction
    for i in range(len(tokens)):
        if tokens[i] == constants.UNK_WORD:
            _, max_index = attn[i].max(0)
            tokens[i] = src_raw[max_index.item()]

    return tokens

def get_temp_and_identi_bpe(prediction_dict,vocab_sep,vocab_token,src_vocabs=None):

    stack_identi=prediction_dict['stack_identi']

    identi_prediction=[]
    for idx,i in enumerate(stack_identi):
        pred=[]
        for j in i:
            if j != constants.EOS:
                pred.append(int(j))
            else:
                break
        res=vocab_token.decode(pred,skip_special_tokens=True)
        identi_prediction.append([i[0] for i in vocab_token.pre_tokenizer.pre_tokenize_str(res)])
    return identi_prediction

def get_temp_and_identi(prediction_dict,vocab_sep,vocab_identi,src_vocabs=None):
    stack_template=prediction_dict['stack_template']
    stack_identi=prediction_dict['stack_identi']

    identi_prediction=[]
    for idx,i in enumerate(stack_identi):
        pred=[]
        for j in i:
            if j != constants.EOS:
                if j < len(vocab_identi):
                    pred.append(vocab_identi[int(j)])
                elif src_vocabs:
                    j = j - len(vocab_identi)
                    pred.append(src_vocabs[idx][int(j)])
                    print("copy{}".format(src_vocabs[idx][int(j)]))
            else:
                break
        identi_prediction.append(pred)



    template_prediction=[]
    for idx,i in enumerate(stack_template):
        pred=[]
        for j in i:
            if j != constants.EOS:
                pred.append(vocab_sep[int(j)])
            else:
                break
        template_prediction.append(pred)
    text=identi_prediction
    return template_prediction,identi_prediction,text



def convert_temp_and_identi(prediction_dict,vocab_sep,vocab_identi):
    stack_template=prediction_dict['stack_template']
    stack_identi=prediction_dict['stack_identi']
    mask_number=(stack_template==4).count_nonzero(dim=1)
    identi_prediction=[]
    for idx,i in enumerate(stack_identi):
        pred=[]
        for j in i:
            if j != constants.EOS:
                pred.append(vocab_identi[int(j)])
            else:
                break
        if len(pred)==0:
            pred=["<unk>"]
        repeat=0
        while len(pred)<mask_number[idx]:
            pred.append(pred[repeat])
            repeat=repeat+1
            if repeat==len(pred):
                repeat=0
        if len(pred)>mask_number[idx]:
            pred=pred[:mask_number[idx]]
        identi_prediction.append(pred)



    template_prediction=[]
    text=[]
    for idx,i in enumerate(stack_template):
        txt=[]
        number_mask=0
        for jdx,j in enumerate(i):
            if j == constants.EOS:
                template_prediction.append(i[:jdx])
                break
            if j == constants.MASK:
                txt.append(identi_prediction[idx][number_mask])
                number_mask=number_mask+1
            else:
                txt.append(vocab_sep[int(j)])
        text.append(txt)


    return template_prediction,identi_prediction,text


def get_its_seq(tp_pred,vocab):
    preds=[]
    for idx, i in enumerate(tp_pred):
        pred=[]
        for j in i:
            if j != constants.EOS:
                pred.append(vocab[int(j)])
            else:
                break
        preds.append(pred)
    return preds