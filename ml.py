import os
import sys
import yaml
import time
import argparse
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics._plot.base import _check_classifer_response_method as resp_func

from sklearn.metrics import precision_recall_curve

from dataset import fetch_bunch, write_predictions
from models import fetch_model


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run final result of text classification."
    )
    parser.add_argument(
        "-d", "--data", type=str,
        default="data/text_cls",
        help="Specify the directory for 'train.bunch', 'val.bunch' data files."
    )
    parser.add_argument(
        "-c", "--config", type=str,
        default="./protocol.yml",
        help="Specify the config file to run the process."
    )
    parser.add_argument(
        "-r", "--record", type=str,
        default="runs",
        help="Specify the directory for records."
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose",
        action="store_true",
        help="Whether to show some of the estimations for val samples."
    )
    parser.add_argument(
        "-i", "--image", dest="image",
        action="store_true",
        help="Whether to show the P-R Curve of this model."
    )
    parser.set_defaults(verbose=False)
    parser.set_defaults(image=False)
    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()
    cfg = arg.config

    train_path = os.path.join(arg.data, "train.bunch")
    val_data = os.path.join(arg.data, "val.txt")
    val_path = os.path.join(arg.data, "val.bunch")

    with open(cfg) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    print("Fitting training data...")
    train_bunch = fetch_bunch(train_path)
    model_cfg = config["model"]
    model_name = model_cfg.get("name", None)
    model_cfg.pop("name")

    # customize the base estimator for AdaBoost Decision Tree.
    if model_name in ["AdaBoostDT"]:
        model_cfg.update(
            {"base_estimator": DecisionTreeClassifier(
                max_depth=4, max_features=1024
            )})

    clf = fetch_model(model_name)(**model_cfg)
    clf.fit(train_bunch.tdm, train_bunch.labels)
    print("Done.")

    val_bunch = fetch_bunch(val_path)
    estimated = clf.predict(val_bunch.tdm)
    respond_func = resp_func(clf, "auto")
    est_prob = respond_func(val_bunch.tdm)
    if est_prob.ndim != 1 and est_prob.shape[1] == 2:
        est_prob = est_prob[:, 1]

    time_format = "%Y-%m-%d...%H.%M.%S"
    id = time.strftime(time_format, time.localtime(time.time()))
    record_path = os.path.join(arg.record, model_name, id)

    os.makedirs(record_path)
    sys.stdout = Logger(os.path.join(record_path, 'records.txt'))

    print("\nResults begin...")

    if arg.verbose:
        i = 0
        for raw, gt, est in zip(val_bunch.raws, val_bunch.labels, estimated):
            if i % 200 == 0:
                print("raw: ", raw, "\ngt: ", gt, "est: ", est)
            i += 1

    print("*" * 25, " RESULTS BEGINS ", "*" * 25)
    print("Model name: %s." % model_name)
    print("Config: %s.\n" % model_cfg)
    p = precision(val_bunch.labels, estimated, average="weighted")
    r = recall(val_bunch.labels, estimated, average="weighted")
    f1_ = f1(val_bunch.labels, estimated, average="weighted")
    print("Precision: %.4f" % p)
    print("Recall: %.4f" % r)
    print("Precision: %.4f" % f1_)
    print("*" * 25, " RESULTS ENDS ", "*" * 25)
    print("Done.\n")

    # output the final results to disk
    write_predictions(
        val_data, os.path.join(record_path, "results.txt"),
        val_bunch.data_indices, estimated
    )

    values = precision_recall_curve(val_bunch.labels, est_prob)
    with open(os.path.join(record_path, "pr.values"), "wb") as f:
        pickle.dump(values, f)
    p_value, r_value, _ = values

    if arg.image:
        plt.figure()
        plt.plot(
            p_value, r_value,
            label="%s (AP: %.2f, F1: %.2f)" % (model_name, p, f1_)
        )
        plt.legend(loc="best")
        plt.title("2-Classes P-R curve")
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.savefig(os.path.join(record_path, "P-R.png"))
        plt.show()
