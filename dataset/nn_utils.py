import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

PAD = "[PAD]"
CLS = "[CLS]"
# Convert raw label to the expected one.
LABEL_MAP = {0: 0, 1: 1}


class ChineseTextSet(Dataset):
    def __init__(self, path, tokenizer, pad, device=None):
        super(ChineseTextSet, self).__init__()
        self.device = torch.device("cpu") if device is None else device
        self.contents = self.prepare_data(
            path, tokenizer, pad, delimiter="\t", apply_func=self._to_tensor)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        return self.contents[item]

    def _to_tensor(self, data):
        def to_LongTensor(obj, d):
            return torch.tensor(obj, dtype=torch.int64, device=d)

        return [to_LongTensor(_, self.device) for _ in data]

    @staticmethod
    def prepare_data(path, tokenizer, pad, delimiter="|", apply_func=lambda x: x):
        """
        Convert the raw chinese text into appropriate format for deep models.

        Args:
            path: The file location where you store your raw text.
            tokenizer: The tokenizer used to generate the tokens.
            pad: The longest length for a sentence while processing.
            delimiter: The delimiter to separate different attributes in the raw
                text file.
            apply_func: The function to apply during processing.

        Returns:
            Processed chinese text data that can be input to deep models.
        """
        print("Preparing data from %s..." % path)
        contents = list()
        with open(path, 'r', encoding="utf-8") as f:
            line = f.readline()
            while line:
                # Specify your data format here. #
                data_idx, label, ctx = line.split(delimiter)

                # You may not need to modify the codes below.
                label = LABEL_MAP[int(label)]
                token = [CLS] + tokenizer.tokenize(ctx)
                token_len = len(token)
                token_ids = tokenizer.convert_tokens_to_ids(token)
                mask = list()
                if pad is not None:
                    if len(token) < pad:
                        mask = [1] * len(token_ids) + [0] * (pad - len(token))
                        token_ids += [0] * (pad - len(token))
                    else:
                        mask = [1] * pad
                        token_ids = token_ids[:pad]
                        token_len = pad
                sample = [token_ids, int(label), token_len, mask, int(data_idx)]
                sample = apply_func(sample)
                contents.append(sample)
                line = f.readline()
        print("Done.")
        return contents


if __name__ == '__main__':
    stop = 5
    batch_size = 100
    data_path = "./path/to/data"
    pretrained_weights = "bert-base-chinese"
    bert_tkzer = BertTokenizer.from_pretrained(pretrained_weights)
    dataset = ChineseTextSet(data_path, bert_tkzer, 32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    def test_dataset():
        for i, _ in enumerate(dataset):
            if i == stop:
                break
            assert len(_) == 5, "The length of datapoint is wrong." \
                                " Expect: 5, but found: %d." % len(_)
            idx, label, length, mask, qid = _
            print("id: ", idx.shape, " label: ", label, end=" ")
            print("length: ", length, " mask: ", mask.shape, " qid: ", qid)


    def test_dataloader():
        for i, _ in enumerate(dataloader):
            if i == stop:
                break
            assert len(_) == 5, "The length of datapoint is wrong." \
                                " Expect: 5, but found: %d." % len(_)
            idx, label, length, mask, qid = _
            print("id: ", idx.shape, " label: ", label.shape, end=" ")
            print("length: ", length.shape, " mask: ", mask.shape, end=" ")
            print("qid: ", qid.shape, " device: ", idx.device)


    test_dataset()
    test_dataloader()
