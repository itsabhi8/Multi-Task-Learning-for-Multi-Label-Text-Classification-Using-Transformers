
from transformers import AutoConfig,AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

#nINF = -100



class LMEnc(nn.Module):
    
    def __init__(self,config,head,backbone='bert-base-uncased',):
        super(LMEnc, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone, return_dict=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.head=head
        if self.head:
            self.fc = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size,),
                nn.ReLU(inplace=True)
            )

    
    def forward(self,input_ids, attention_mask):

        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state']
        bert_output = self.dropout(bert_output)
        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L
        attention = self.attention(bert_output).transpose(1, 2).masked_fill(~masks, -np.inf)  # N, labels_num, L
        attention = F.softmax(attention, -1)
        # attention = Sparsemax(dim=-1)(attention)
        representation = attention @ bert_output   # N, labels_num, hidden_size
        if self.head:
            representation= self.fc(representation)

        return representation, self.attention.weight




class PLM_MTC(nn.Module):
    def __init__(self,config,num_labels,backbone,head,bce_wt):
        super(PLM_MTC, self).__init__()


        self.num_labels=num_labels
        config.num_labels=num_labels
        self.bce_wt=bce_wt
        self.textenc=LMEnc(config,head,backbone)

        self.classifier = nn.Linear(num_labels*config.hidden_size, num_labels)
        #self.classifier2 = nn.Linear(num_labels * 768,
        #                        num_labels)

        self.dml=0

        
        
    def forward(self, input_ids, attention_mask,labels):
        output,label_emb = self.textenc(input_ids, attention_mask)
        logits=self.classifier(output.view(output.shape[0],-1))
        
        loss=0
        
        if self.training:
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)
                loss += loss_fct(logits.view(-1, self.num_labels), target)*(self.bce_wt)
            
            if self.dml:
                pass
                               

        

        return {
            'loss': loss,
            'logits': logits,
            #'attentions': attns,
            #'hidden_states': label_aware_embedding,
            #'contrast_logits': contrast_logits,
            }
        






