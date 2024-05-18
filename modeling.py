# coding=utf-8

import os
import pdb
import copy
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import CrossEntropyLoss
from transformers import (
    BertConfig,
    BertModel,
    RobertaModel,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer
)


class BERTMultiNER2(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(BERTMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Create a general NER classfier
        self.gener_classifier = torch.nn.Linear(
            config.hidden_size, self.num_labels)

        # create a bio NER classfier
        self.bio_classifier = torch.nn.Linear(
            config.hidden_size, self.num_labels)

        self.bio_classifier_2 = torch.nn.Linear(
            config.hidden_size, config.hidden_size)

        self.gener_classifier_2 = torch.nn.Linear(
            config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            gener_sequence_output = F.relu(self.gener_classifier_2(
                sequence_output))  # general NER logit value
            bio_sequence_output = F.relu(self.bio_classifier_2(
                sequence_output))  # Bio logit value

            gener_logits = self.gener_classifier(
                gener_sequence_output)  # general NER logit value
            bio_logits = self.bio_classifier(
                bio_sequence_output)  # Bio logit value

            # update logit and sequence_output
            sequence_output = bio_sequence_output + gener_sequence_output
            logits = (bio_logits, gener_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            bio_idx = copy.deepcopy(entity_type_ids)
            gener_idx = copy.deepcopy(entity_type_ids)

            # BioNER index range is 1-9
            bio_idx[(bio_idx == 0) | (bio_idx > 9)] = 0
            gener_idx[gener_idx != 0] = 0  # General NER index is 0

            bio_sequence_output = bio_idx.unsqueeze(-1) * sequence_output
            gener_sequence_output = gener_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu

            bio_sequence_output = F.relu(self.bio_classifier_2(
                sequence_output))  # Bio logit value
            gener_sequence_output = F.relu(self.gener_classifier_2(
                gener_sequence_output))  # general NER logit value

            bio_logits = self.bio_classifier(
                bio_sequence_output)  # Bio logit value
            gener_logits = self.gener_classifier(
                gener_sequence_output)  # generic NER logit value

            # update logit and sequence_output
            sequence_output = bio_sequence_output + gener_sequence_output
            logits = bio_logits + gener_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if entity_type_ids[0][0].item() == 0:
                    active_loss = attention_mask.view(-1) == 1
                    bio_logits, gener_logits = logits

                    active_bio_logits = bio_logits.view(-1, self.num_labels)
                    active_gener_logits = gener_logits.view(
                        -1, self.num_labels)

                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(
                            loss_fct.ignore_index).type_as(labels)
                    )
                    bio_loss = loss_fct(active_bio_logits, active_labels)
                    gener_loss = loss_fct(active_gener_logits, active_labels)

                    loss = bio_loss + gener_loss

                    return ((loss,) + outputs)
                else:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(
                            loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                    return ((loss,) + outputs)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class RoBERTaMultiNER2(RobertaForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(RoBERTaMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Create a general NER classfier
        self.gener_classifier = torch.nn.Linear(
            config.hidden_size, self.num_labels)  # general NER

        # create a bio NER classfier
        self.bio_classifier = torch.nn.Linear(
            config.hidden_size, self.num_labels)

        self.bio_classifier_2 = torch.nn.Linear(
            config.hidden_size, config.hidden_size)

        self.gener_classifier_2 = torch.nn.Linear(
            config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.roberta(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            gener_sequence_output = F.relu(self.gener_classifier_2(
                sequence_output))  # general NER logit value
            bio_sequence_output = F.relu(self.bio_classifier_2(
                sequence_output))  # Bio logit value

            gener_logits = self.gener_classifier(
                gener_sequence_output)  # general NER logit value
            bio_logits = self.bio_classifier(
                bio_sequence_output)  # Bio logit value

            # update logit and sequence_output
            sequence_output = bio_sequence_output + gener_sequence_output
            logits = (bio_logits, gener_logits)
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            bio_idx = copy.deepcopy(entity_type_ids)
            gener_idx = copy.deepcopy(entity_type_ids)

            # BioNER index range is 1-9
            bio_idx[(bio_idx == 0) | (bio_idx > 9)] = 0
            gener_idx[gener_idx != 0] = 0  # General NER index is 0

            bio_sequence_output = bio_idx.unsqueeze(-1) * sequence_output
            gener_sequence_output = gener_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu

            bio_sequence_output = F.relu(self.bio_classifier_2(
                sequence_output))  # Bio logit value
            gener_sequence_output = F.relu(self.gener_classifier_2(
                gener_sequence_output))  # general NER logit value

            bio_logits = self.bio_classifier(
                bio_sequence_output)  # Bio logit value
            gener_logits = self.gener_classifier(
                gener_sequence_output)  # generic NER logit value

            # update logit and sequence_output
            sequence_output = bio_sequence_output + gener_sequence_output
            logits = bio_logits + gener_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if entity_type_ids[0][0].item() == 0:
                    active_loss = attention_mask.view(-1) == 1
                    bio_logits, gener_logits = logits

                    active_bio_logits = bio_logits.view(-1, self.num_labels)
                    active_gener_logits = gener_logits.view(
                        -1, self.num_labels)

                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(
                            loss_fct.ignore_index).type_as(labels)
                    )
                    bio_loss = loss_fct(active_bio_logits, active_labels)
                    gener_loss = loss_fct(active_gener_logits, active_labels)

                    loss = bio_loss + gener_loss

                    return ((loss,) + outputs)
                else:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(
                            loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                    return ((loss,) + outputs)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class NER(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(NER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(
            input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(
                        loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits
