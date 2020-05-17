import os
import jieba
import argparse

from dataset.utils import word_segment, vectorize


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the raw data for text classification."
    )
    parser.add_argument(
        "-b", "--byproduct", type=str, required=True,
        help="Specify the directory for storing by-product files."
    )
    parser.add_argument(
        "-p", "--product", type=str, required=True,
        help="Specify the directory for storing product files for training and validation."
    )
    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()
    byprod = arg.byproduct
    prod = arg.product
    word_segment(
        "./path/to/data", os.path.join(byprod, "stopwords.txt"),
        jieba.cut, target_dir=byprod)
    word_segment(
        "./path/to/data", os.path.join(byprod, "stopwords.txt"),
        jieba.cut, split="val", target_dir=byprod)

    train_bunch = vectorize(seg_dir=byprod, split="train", store_dir=prod)
    vectorize(seg_dir=byprod, split="val", voc=train_bunch.voc, store_dir=prod, verbose=False)
