import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

"""Bert Model transformer with multiple independent binary sequence classification heads on top (a linear layer on top of
the pooled output) to implement a multi-label classification in a "binary-relevance" fashion:
"""
class BertForBRSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # binary classifier per each label
        self.classifiers = [nn.Linear(config.hidden_size, 2)] * self.num_labels
        for i, c in enumerate(self.classifiers):
            self.add_module(f'classifier-class-{i+1}', c)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.BoolTensor` of shape :obj:`(batch_size,num_labels)`, `optional`):
            Labels for computing the sequence classification/regression loss, in multi-hot vectors.
            Returned loss is the sum of each label's regression loss (cross-entropy loss, resp.)
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        label_binary_logits = torch.stack([c(pooled_output) for c in self.classifiers])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            loss = torch.stack([loss_fct(logits, labels) for (logits, labels) in zip(label_binary_logits, labels.long().permute(1,0))])

        output = (label_binary_logits.permute(1,0,2),) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
