import torch
import torch.nn as nn

from component.utils.misc import aeq
from component.decoders.state import RNNDecoderState
from component.modules.global_attention import GlobalAttention


class DecoderBase(nn.Module):
    """Abstract class for decoders.
    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.
        Subclasses should override this method.
        """

        raise NotImplementedError
