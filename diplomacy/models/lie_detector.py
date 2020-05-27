import torch

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.metrics import F1Measure
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

from typing import Optional, Dict

@Model.register('lie_detector')
class LieDetector(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 posclass_weight: Optional[float] = 1,
                 use_power: Optional[bool] = False,
                 dropout: Optional[float] = 0) -> None:
        super().__init__(vocab)
        
        self.embedder = embedder
        self.encoder = encoder
        if use_power:
            self.classifier = torch.nn.Linear(
                in_features=encoder.get_output_dim() + 1,
                out_features=vocab.get_vocab_size('labels')
            )
        else:
            self.classifier = torch.nn.Linear(
                in_features=encoder.get_output_dim(),
                out_features=vocab.get_vocab_size('labels')
            )
        self.use_power = use_power
    
        self.f1_lie = F1Measure(vocab.get_token_index('False', 'labels'))
        self.f1_truth = F1Measure(vocab.get_token_index('True', 'labels'))
        self.micro_f1 = FBetaMeasure(average='micro')
        self.macro_f1 = FBetaMeasure(average='macro')
        
        weights = [1,1]
        weights[vocab.get_token_index('False', 'labels')] = posclass_weight        
        self.loss = torch.nn.CrossEntropyLoss(weight = torch.Tensor(weights))

        self.dropout = torch.nn.Dropout(dropout)
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        lie_precision, lie_recall, lie_fscore = self.f1_lie.get_metric(reset)
        truth_precision, truth_recall, truth_fscore = self.f1_truth.get_metric(reset)
        micro_metrics = self.micro_f1.get_metric(reset)
        macro_metrics = self.macro_f1.get_metric(reset)

        return {
            'truth_precision': truth_precision,
            'truth_recall': truth_recall,
            'truth_fscore': truth_fscore,
            'lie_precision': lie_precision,
            'lie_recall': lie_recall,
            'lie_fscore': lie_fscore,
            'macro_fscore': macro_metrics['fscore'],
            'micro_precision':micro_metrics['precision'], 
            'micro_recall':micro_metrics['recall'], 
            'micro_fscore':micro_metrics['fscore']            
        }

    def forward(self,
                message: Dict[str, torch.Tensor],                
                score_delta: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(message)
        embedded = self.embedder(message)
        #embedded = self._dropout(embedded)        

        encoded = self.encoder(embedded, mask)
        if self.use_power:
            encoded = torch.cat((score_delta.view(-1,1),encoded),1)         
        encoded = self.dropout(encoded)
        
        classified = self.classifier(encoded)

        output = {}
        output["logits"] = classified
        if label is not None:
            self.f1_lie(classified, label)
            self.f1_truth(classified, label)
            self.micro_f1(classified, label)
            self.macro_f1(classified, label)            
            output["loss"] = self.loss(classified, label)
        
        return output
