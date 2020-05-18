from typing import Iterator, List, Dict, Tuple, Optional

from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, LabelField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import MetadataField
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

import numpy as np
import json

@DatasetReader.register('message_reader')
class MessageReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 sender_annotation: bool = True) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}        
        self.tokenizer = tokenizer or WordTokenizer()
        self.sender_annotation = sender_annotation

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as fp:
            for line in fp:
                text = json.loads(line)
                if self.sender_annotation:
                    if text['sender_annotation'] != 'NOANNOTATION':
                            yield self.text_to_instance(
                                text['message'],
                                text['score_delta'],
                                label=str(text['sender_annotation'])
                        )
                else:
                    if text['receiver_annotation'] != 'NOANNOTATION':                        
                        yield self.text_to_instance(
                            text['message'],
                            text['score_delta'],
                            label=str(text['receiver_annotation'])
                    )

    def text_to_instance(self, message: str, score_delta: int, 
                            label: List[str]=None) -> Field:
        instance_fields: Dict[str, Field] = {}
        instance_fields['message'] = TextField(self.tokenizer.tokenize(message), self._token_indexers)
        instance_fields['score_delta'] = ArrayField(np.array(score_delta))
        if label is not None:
            instance_fields['label'] = LabelField(label)
        
        return Instance(instance_fields)
