import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

PRETRAINED_WEIGHTS = "roberta-base"


class RoBERT(nn.Module):
    def __init__(self, weights=None, hidden_state=768, num_classes=2):
        super(RoBERT, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
        if weights is None:
            # for training
            self.bert = RobertaModel.from_pretrained(PRETRAINED_WEIGHTS)
        else:
            # for testing
            self.bert = RobertaModel.from_pretrained(None, **weights)
        self.config = self.bert.config
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_state, num_classes)

    def forward(self, x):
        ctx, mask = x
        _, pooled = self.bert(ctx, attention_mask=mask)
        return self.fc(pooled)


if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    model = RobertaModel.from_pretrained(PRETRAINED_WEIGHTS)
