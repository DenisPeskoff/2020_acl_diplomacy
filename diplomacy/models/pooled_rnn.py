import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from typing import Optional, Dict, Any, List, Union, Tuple

@Seq2VecEncoder.register('pooled_rnn')
class PooledRNN(Seq2VecEncoder):
    """
    A base-to-higher-level module.
    """
    def __init__(self,
                encoder: Seq2SeqEncoder,
                poolers: str = "max,mean,last") -> None:
        super(PooledRNN, self).__init__()
        self._encoder = encoder
        self._poolers = poolers.split(',')

    def forward(self, matrix: torch.Tensor, matrix_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs: pack_padded_sequence of (batch, max_length, input_size)
        Outpus: (batch, hidden_size*2)
        """
        # run through the bidirectional RNN
        # shape : batch, sequence, embeddings
        encoded = self._encoder(matrix, matrix_mask)
        # print(encoded.shape)
        batch_size, sequence_length, encoding_dim = encoded.shape
        mask = matrix_mask.bool()
        lengths = get_lengths_from_binary_sequence_mask(mask)
        pooled = []

        if 'max' in self._poolers:
            max_pool = encoded.masked_fill(~mask[:, :, None], -float('inf')).max(dim=1)[0]
            pooled.append(max_pool)

        if 'mean' in self._poolers:
            avg_pool = encoded.sum(1)
            # Set any length 0 to 1, to avoid dividing by zero.
            lengths = torch.max(lengths, lengths.new_ones(1))
            avg_pool = avg_pool / lengths.unsqueeze(-1).float()
            pooled.append(avg_pool)

        if 'last' in self._poolers:
            indexes = lengths - 1
            indexes = indexes.view(-1, 1).expand(len(lengths), encoded.size(2))
            indexes = indexes.unsqueeze(1)
            last_state = encoded.gather(1, indexes).squeeze(1)

            if not self._encoder._module.bidirectional:
                pooled.append(last_state)
            else:
                first_state = encoded[:, 0, self._encoder.get_output_dim():]
                last_state = last_state[:, :self._encoder.get_output_dim()]
                bidirectional_states = torch.cat([first_state, last_state], dim=1)
                pooled.append(bidirectional_states)
                
        return torch.cat(pooled, dim=1)

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        return self._encoder.get_input_dim()

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._encoder.get_output_dim() * len(self._poolers)