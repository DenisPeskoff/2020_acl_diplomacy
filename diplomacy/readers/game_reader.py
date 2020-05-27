from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from overrides import overrides
from typing import Dict, List, Iterator, Any, Optional

import numpy as np
import json


@DatasetReader.register("diplomacy_reader")
class DiplomacyReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 use_game_scores: bool = False,
                 label_key: str = 'sender_labels') -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer()
        self._use_game_scores = use_game_scores
        self._label_key = label_key

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as game_file:
            for conversation in game_file:
                instance = json.loads(conversation)
                #print(instance.keys())
                # => dict_keys(['messages', 'sender_labels', 'receiver_labels', 'speakers', 'receivers', 'game_score', 'absolute_message_index', 'relative_message_index', 'seasons', 'years'])
                if len(instance['messages']) > 0:
                    instance = self.text_to_instance(
                        instance['messages'],
                        instance['speakers'],
                        instance[self._label_key],
                        instance['game_score_delta'])
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(self,
                         messages: List[str],
                         speakers: List[str],
                         labels: List[str],
                         game_score_delta: Optional[List[int]] = None) -> Instance:
        fields: Dict[str, Field] = {}

        nm, ns, nls, ng = [], [], [], []
        for m, s, ls, g in zip(messages, speakers, labels, game_score_delta):
            if ls not in [True, False, 'true', 'false', 'True', 'False']:
                continue
            nm.append(m) ; ns.append(s) ; nls.append(ls) ; ng.append(g)

        messages, speakers, labels, game_score_delta = nm, ns, nls, ng
        if len(messages) == 0: return None

        if self._use_game_scores and game_score_delta is not None:
            game_score_delta_field = ArrayField(np.array(game_score_delta))
            fields['game_scores'] = game_score_delta_field

        # wrap each token in the file with a token object
        messages_field = ListField([
            TextField(self._tokenizer.tokenize(message), self._token_indexers)
                for message in messages])

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["messages"] = messages_field
        #fields["speakers"] = SequenceLabelField(speakers, messages_field)
        labels = SequenceLabelField([str(l) for l in labels], messages_field)

        fields["labels"] = labels

        return Instance(fields)
