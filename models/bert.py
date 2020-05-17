import torch.nn as nn
from transformers import BertTokenizer, BertModel

PRETRAINED_WEIGHTS = "bert-base-chinese"


class BERT(nn.Module):
    def __init__(self, weights=None, hidden_state=768, num_classes=2):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
        if weights is None:
            # for training
            self.bert = BertModel.from_pretrained(PRETRAINED_WEIGHTS)
        else:
            # for testing
            self.bert = BertModel.from_pretrained(None, **weights)
        self.config = self.bert.config
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_state, num_classes)

    def forward(self, x):
        ctx, mask = x
        _, pooled = self.bert(ctx, attention_mask=mask)
        return self.fc(pooled)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    model = BertModel.from_pretrained(PRETRAINED_WEIGHTS)
