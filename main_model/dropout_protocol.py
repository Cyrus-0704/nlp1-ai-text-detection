"""Classifier-only dropout intervention; never changes encoder dropout or weights."""
import torch


def apply_classifier_dropout(model,value,resuming=False):
    if value!=.2:raise ValueError('This reviewed intervention is classifier dropout0.2 only')
    if model.config.model_type!='xlm-roberta':raise ValueError('Expected XLM-R')
    if model.config.hidden_dropout_prob!=.1 or model.config.attention_probs_dropout_prob!=.1:
        raise ValueError('Encoder/attention must remain0.1')
    if resuming and model.config.classifier_dropout!=value:
        raise ValueError('Cannot resume a0.1 trajectory as0.2')
    if any(m.p!=.1 for m in model.roberta.modules() if isinstance(m,torch.nn.Dropout)):
        raise ValueError('Unexpected encoder dropout module')
    model.config.classifier_dropout=value
    model.classifier.dropout.p=value
    return dict(classifier=value,hidden=.1,attention=.1,scope='classifier dropout module (used twice by XLM-R head)')
