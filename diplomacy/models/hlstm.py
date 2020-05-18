import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.metrics import FBetaMeasure
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from typing import Dict, Optional


def weighted_sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                                  targets: torch.LongTensor,
                                                  mask: torch.FloatTensor,
                                                  weights: torch.FloatTensor = None) -> torch.FloatTensor:
    num_classes = logits.size(-1)
    # make sure weights are float
    mask = mask.float().view(-1)
    logits = logits.view(-1, num_classes)
    targets = targets.view(-1)
    #print(mask.shape, logits.shape, targets.shape)
    loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    #print(loss.shape)
    loss = (loss * mask).mean()
    #print('')
    return loss


@Model.register('hierarchical_lstm')
class HierarchicalLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 message_encoder: Seq2VecEncoder,
                 conversation_encoder: Seq2SeqEncoder,
                 dropout: float = 0.5,
                 pos_weight: float = None,
                 use_game_scores: bool = False) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._message_encoder = message_encoder
        self._conversation_encoder = conversation_encoder
        self._use_game_scores = use_game_scores

        output_dim = conversation_encoder.get_output_dim() + int(self._use_game_scores)

        self._classifier = nn.Linear(in_features=output_dim,
                                     out_features=vocab.get_vocab_size('labels'))
        self._dropout = nn.Dropout(dropout)

        self._label_index_to_token = vocab.get_index_to_token_vocabulary(namespace="labels")
        self._num_labels = len(self._label_index_to_token)
        print(self._label_index_to_token)
        index_list = list(range(self._num_labels))
        print(index_list)
        self._f1 = FBetaMeasure(average=None, labels=index_list)
        self._f1_micro = FBetaMeasure(average='micro')
        self._f1_macro = FBetaMeasure(average='macro')

        if pos_weight is None or pos_weight <= 0:
            labels_counter = self.vocab._retained_counter['labels']
            self._pos_weight = 1. * labels_counter['True'] / labels_counter['False']
            # self._pos_weight = 15.886736214605067
            print('Computing Pos weight from labels:', self._pos_weight)
        else:
            self._pos_weight = float(pos_weight)

    def forward(self,
                messages: Dict[str, torch.Tensor],
                labels: Optional[torch.Tensor] = None,
                game_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(messages, num_wrapping_dims=1)
        conversation_mask = (mask.sum(dim=-1) > 0).float()

        embedded = self._embedder(messages)
        embedded = self._dropout(embedded)

        batch, num_messages, num_tokens, embedding_dim = embedded.shape

        embedded_reshaped = embedded.view(batch * num_messages, num_tokens, embedding_dim)
        mask_reshaped = mask.view(batch * num_messages, num_tokens)

        encoded_messages = self._message_encoder(embedded_reshaped, mask_reshaped)
        encoded_messages = self._dropout(encoded_messages)
        encoded_messages = encoded_messages.view(batch, num_messages, -1)

        encoded_conversation = self._conversation_encoder(encoded_messages, conversation_mask)

        if self._use_game_scores and game_scores is not None:
            encoded_conversation = torch.cat([encoded_conversation, game_scores.view(batch, num_messages, 1)], dim=2)

        encoded_conversation = self._dropout(encoded_conversation)

        classified = self._classifier(encoded_conversation)

        output: Dict[str, torch.Tensor] = {}

        if labels is not None:
            #output["loss"] = sequence_cross_entropy_with_logits(classified, labels, conversation_mask, gamma=100.0)

            output["loss"] = weighted_sequence_cross_entropy_with_logits(
                classified,
                labels,
                conversation_mask,
                weights = torch.Tensor([1.0, self._pos_weight]).cuda())
            output['prediction'] = classified

            self._f1(classified, labels, conversation_mask)
            self._f1_micro(classified, labels, conversation_mask)
            self._f1_macro(classified, labels, conversation_mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self._label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}

        return {**f1_scores,
                **f1_micro,
                **f1_macro}
