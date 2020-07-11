import _pickle as pickle
import os
from pprint import pprint

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(666)


class RawDataset(object):
    def __init__(self):
        self.data = list()

    def append(self, raw, context, label, data_id):
        self.data.append([raw, context, label, data_id])

    def shuffle(self):
        np.random.shuffle(self.data)

    @property
    def data_indices(self):
        return [_[3] for _ in self.data]

    @property
    def raws(self):
        return [_[0] for _ in self.data]

    @property
    def contents(self):
        return [_[1] for _ in self.data]

    @property
    def labels(self):
        return [_[2] for _ in self.data]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def combine_stopwords(source_path, target_path):
    """
    The function is used to combine 2 stopwords lists.

    Args:
        source_path: list, specifies the source path.
        target_path: str, specifies the target path.

    Returns:
        The aggregated stopwords lists.
    """
    stopwords = set()
    print("Reading source files...")
    for p in source_path:
        file = open(p, "r", encoding="utf-8")
        word = file.readline()
        while word:
            stopwords.add(word)
            word = file.readline()
        file.close()
    print("Reading files done.")

    print("\nWriting to target files...")
    with open(target_path, "w", encoding="utf-8") as f:
        for _ in stopwords:
            f.writelines(_)
    print("Done.")


def fetch_stopwords(file_path):
    """
    Fetch stopwords lists.

    Args:
        file_path: str.

    Returns:
        The stopwords lists.
    """
    sw = set()
    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            sw.add(line.replace("\n", ""))
            line = f.readline()
    return sw


def word_segment(file_path, stop_words, seg_func, delimiter="\t",
                 skip_first=False, target_dir=None, split="train"):
    """
    Perform word segment and (optionally) store the results.

    Args:
        file_path: str, the raw dataset path.
        stop_words: list or str,
            if list: all stopwords in lists;
            if str: stopwords file path.
        seg_func: func, the function used to segment words.
        delimiter: str, delimiter of different fields in a sample.
        skip_first: bool, whether to skip the first line of the
            input file.
        target_dir: if not None, store the results to given path.
        split: str, 'train' or 'val'.

    Returns:
        A dictionary with keys 0 and 1 specify the negative samples and
        positive samples in the type of list.
    """
    sw = set()
    if stop_words is None:
        pass
    elif isinstance(stop_words, str):
        with open(stop_words, "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                sw.add(line.replace("\n", ""))
                line = f.readline()

    text = {0: [], 1: []}
    label_map = {0: 0, 1: 1}
    print("Processing word segmentation...")
    data_id = 1
    with open(file_path, "r", encoding="utf-8") as f:
        if skip_first:
            f.readline()
        line = f.readline()
        while line:
            values = line.split(delimiter)
            # word segmentation
            segments = [_ for _ in seg_func(values[-1]) if _ not in sw]
            raw = values[-1].replace("\n", "")
            text.get(label_map[int(values[0])]).append(
                str(data_id) + delimiter + raw + delimiter + " ".join(segments))
            line = f.readline()
            data_id += 1
    print("Done.")

    if target_dir is not None:
        print("\nStoring word segments...")
        neg_path = os.path.join(target_dir, split + "_0.txt")
        with open(neg_path, "w", encoding="utf-8") as f:
            for _ in text.get(0):
                f.writelines(_)
        pos_path = os.path.join(target_dir, split + "_1.txt")
        with open(pos_path, "w", encoding="utf-8") as f:
            for _ in text.get(1):
                f.writelines(_)
        print("Done.")
    return text


def vectorize(seg_dir, delimiter="\t", split="train", min_df=1,
              voc=None, verbose=True, store_dir=None):
    """
    Vectorize the word segment with tf-idf strategy and (optionally) store a
    RawDataset object.

    Args:
        seg_dir: str, the path to the word segment file.
        delimiter: The delimiter to separate different attributes in the raw
            text file.
        split: str, 'train' or 'val'.
        min_df: int or float, see @TfidfVectorizer.min_df.
        voc: dict, the vocabulary, used when the split is 'val'.
        verbose: bool, whether to print the information of the vectorization.
        store_dir: if not None, store the results to given path.

    Returns:
        A RawDataset object.
    """
    bunch = RawDataset()
    # fetch the negative samples.
    with open(os.path.join(seg_dir, split + "_0.txt"), "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            text = line.split(delimiter)
            # text [data_idx, raw, word_segment]
            bunch.append(text[1], text[2], 0, int(text[0]))
            line = f.readline()
    # fetch the positive samples.
    with open(os.path.join(seg_dir, split + "_1.txt"), "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            text = line.split(delimiter)
            # text [data_idx, raw, word_segment]
            bunch.append(text[1], text[2], 1, int(text[0]))
            line = f.readline()
    bunch.shuffle()

    print("Vectorization begin...")
    if split == "train":
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df)
        bunch.tdm = tfidf.fit_transform(bunch.contents)
        bunch.voc = tfidf.vocabulary_
    else:
        tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=voc)
        bunch.tdm = tfidf.fit_transform(bunch.contents)
        bunch.voc = voc
    print("Done.")

    if verbose:
        print("\nTDM Matrix shape: {}.".format(bunch.tdm.shape))
        print("The 10 highest frequent words:")
        pprint(sorted(bunch.voc.items(), key=lambda kv: (
            kv[1], kv[0]), reverse=True)[:10])

    if store_dir is not None:
        print("\nStoring bunch object...")
        with open(os.path.join(store_dir, split + ".bunch"), "wb") as f:
            pickle.dump(bunch, f)
        print("Done.")
    return bunch


def write_predictions(from_path, to_path, data_ids, pred, delimiter="\t", skip_first=False):
    print("Writing predictions as 'txt' file to '%s'..." % to_path)
    results = dict()
    data_id = 1
    with open(from_path, "r", encoding="utf-8") as f:
        if skip_first:
            f.readline()
        line = f.readline()
        while line:
            # Specify your data format here. #
            _, _, ctx = line.split(delimiter)

            results.update({int(data_id): ctx})
            line = f.readline()
            data_id += 1
    for data_id, est in zip(data_ids, pred):
        if data_id in results:
            results.update({data_id:  delimiter + str(est) + delimiter + results.get(data_id)})
        else:
            raise ValueError(
                "When writing predictions to disk, data id: %d not found in file: %s."
                % (data_id, from_path)
            )
    with open(to_path, "w", encoding="utf-8") as f:
        f.writelines("qid\tlabel\ttext\n")
        for k, v in results.items():
            f.writelines(str(k) + v + "\n")
    print("Done.")
    print("\nPlease refer to '%s' to check the final predictions." % to_path)


def fetch_bunch(file_path):
    with open(file_path, "rb") as f:
        bunch = pickle.load(f)
    return bunch


if __name__ == '__main__':
    base_dir = "path/to/base_dir"
    combine_stopwords(
        source_path=[base_dir + "baidu_stopwords.txt",
                     base_dir + "hit_stopwords.txt"],
        target_path=base_dir + "stopwords.txt"
    )
