import _pickle as pickle
import argparse
import math
import os
import random
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import AdamW

from dataset import ChineseTextSet, PAD, CLS, write_predictions
from losses import fetch_loss
from models import fetch_nn
from result import Logger
from schedulers import fetch_scheduler


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run neural network for text classification."
    )
    parser.add_argument(
        "-s", "--stage", type=str,
        required=True,
        help="Specify the stage for the process, either 'train' or 'test'."
    )
    parser.add_argument(
        "-p", "--proto", type=str,
        default="./bert.yml",
        help="Specify the protocol file to run the process."
    )
    parser.add_argument(
        "-r", "--record", type=str,
        default="runs",
        help="Specify the directory for records."
    )
    parser.add_argument(
        "-i", "--image", dest="image",
        action="store_true",
        help="Whether to show the P-R Curve of this model."
    )
    parser.set_defaults(image=False)
    return parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """The class for timer."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class Trainer(object):
    def __init__(self, proto, stage="train"):
        # model config
        model_cfg = proto["model"]
        model_name = model_cfg["name"]
        self.model_name = model_name

        # dataset config
        data_cfg = proto["data"]
        train_data_path = data_cfg.get("train_path", None)
        val_data_path = data_cfg.get("val_path", None)
        pad = data_cfg.get("pad", 32)
        train_bs = data_cfg.get("train_batch_size", None)
        val_bs = data_cfg.get("val_batch_size", None)
        self.val_bs = val_bs
        self.skip_first = data_cfg.get("skip_first", False)
        self.delimiter = data_cfg.get("delimiter", "\t")

        # assorted config
        optim_cfg = proto.get("optimizer", {"lr": 0.00003})
        sched_cfg = proto.get("schedulers", None)
        loss = proto.get("loss", "CE")
        self.device = proto.get("device", None)

        model_cfg.pop("name")

        if torch.cuda.is_available() and self.device is not None:
            print("Using device: %d." % self.device)
            self.device = torch.device(self.device)
            self.gpu = True
        else:
            print("Using cpu device.")
            self.device = torch.device("cpu")
            self.gpu = False

        if stage == "train":
            if train_data_path is None or val_data_path is None:
                raise ValueError("Please specify both train and val data path.")
            if train_bs is None or val_bs is None:
                raise ValueError("Please specify both train and val batch size.")
            # loading model
            self.model = fetch_nn(model_name)(**model_cfg)
            self.model = self.model.cuda(self.device)

            # loading dataset and converting into dataloader
            self.train_data = ChineseTextSet(
                path=train_data_path, tokenizer=self.model.tokenizer, pad=pad,
                delimiter=self.delimiter, skip_first=self.skip_first)
            self.train_loader = DataLoader(
                self.train_data, train_bs, shuffle=True, num_workers=4)
            self.val_data = ChineseTextSet(
                path=val_data_path, tokenizer=self.model.tokenizer, pad=pad,
                delimiter=self.delimiter, skip_first=self.skip_first)
            self.val_loader = DataLoader(
                self.val_data, val_bs, shuffle=True, num_workers=4)

            time_format = "%Y-%m-%d...%H.%M.%S"
            id = time.strftime(time_format, time.localtime(time.time()))
            self.record_path = os.path.join(arg.record, model_name, id)

            os.makedirs(self.record_path)
            sys.stdout = Logger(os.path.join(self.record_path, 'records.txt'))
            print("Writing proto file to file directory: %s." % self.record_path)
            yaml.dump(proto, open(os.path.join(self.record_path, 'protocol.yml'), 'w'))

            print("*" * 25, " PROTO BEGINS ", "*" * 25)
            pprint(proto)
            print("*" * 25, " PROTO ENDS ", "*" * 25)

            self.optimizer = AdamW(self.model.parameters(), **optim_cfg)
            self.scheduler = fetch_scheduler(self.optimizer, sched_cfg)

            self.loss = fetch_loss(loss)

            self.best_f1 = 0.0
            self.best_step = 1
            self.start_step = 1

            self.num_steps = proto["num_steps"]
            self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

            # the number of steps to write down a log
            self.log_steps = proto["log_steps"]
            # the number of steps to validate on val dataset once
            self.val_steps = proto["val_steps"]

            self.f1_meter = AverageMeter()
            self.p_meter = AverageMeter()
            self.r_meter = AverageMeter()
            self.acc_meter = AverageMeter()
            self.loss_meter = AverageMeter()

        if stage == "test":
            if val_data_path is None:
                raise ValueError("Please specify the val data path.")
            if val_bs is None:
                raise ValueError("Please specify the val batch size.")
            id = proto["id"]
            ckpt_fold = proto.get("ckpt_fold", "runs")
            self.record_path = os.path.join(ckpt_fold, model_name, id)
            sys.stdout = Logger(os.path.join(self.record_path, 'tests.txt'))

            config, state_dict, fc_dict = self._load_ckpt(best=True, train=False)
            weights = {"config": config, "state_dict": state_dict}
            # loading trained model using config and state_dict
            self.model = fetch_nn(model_name)(weights=weights)
            # loading the weights for the final fc layer
            self.model.load_state_dict(fc_dict, strict=False)
            # loading model to gpu device if specified
            if self.gpu:
                self.model = self.model.cuda(self.device)

            print("Testing directory: %s." % self.record_path)
            print("*" * 25, " PROTO BEGINS ", "*" * 25)
            pprint(proto)
            print("*" * 25, " PROTO ENDS ", "*" * 25)

            self.val_path = val_data_path
            self.test_data = ChineseTextSet(
                path=val_data_path, tokenizer=self.model.tokenizer, pad=pad,
                delimiter=self.delimiter, skip_first=self.skip_first)
            self.test_loader = DataLoader(
                self.test_data, val_bs, shuffle=True, num_workers=4)

    def _save_ckpt(self, step, best=False, f=None, p=None, r=None):
        save_dir = os.path.join(self.record_path, "best_model.bin" if best else "latest_model.bin")
        torch.save({
            "step": step,
            "f1": f,
            "precision": p,
            "recall": r,
            "best_step": self.best_step,
            "best_f1": self.best_f1,
            "model": self.model.state_dict(),
            "config": self.model.config,
            "optimizer": self.optimizer.state_dict(),
            "schedulers": self.scheduler.state_dict(),
        }, save_dir)

    def _load_ckpt(self, best=False, train=False):
        load_dir = os.path.join(self.record_path, "best_model.bin" if best else "latest_model.bin")
        load_dict = torch.load(load_dir, map_location=self.device)
        self.start_step = load_dict["step"]
        self.best_step = load_dict["best_step"]
        self.best_f1 = load_dict["best_f1"]
        if train:
            self.optimizer.load_state_dict(load_dict["optimizer"])
            self.scheduler.load_state_dict(load_dict["schedulers"])
        print("Loading checkpoint from %s, best step: %d, best f1: %.4f."
              % (load_dir, self.best_step, self.best_f1))
        if not best:
            print("Checkpoint step %s, f1: %.4f, precision: %.4f, recall: %.4f."
                  % (self.start_step, load_dict["f1"],
                     load_dict["precision"], load_dict["recall"]))
        fc_dict = {
            "fc.weight": load_dict["model"]["fc.weight"],
            "fc.bias": load_dict["model"]["fc.bias"]
        }
        return load_dict["config"], load_dict["model"], fc_dict

    def to_cuda(self, *args):
        return [obj.cuda(self.device) for obj in args]

    @staticmethod
    def fixed_randomness():
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def update_metrics(gt, pre, f1_m, p_m, r_m, acc_m):
        f1_value = f1(gt, pre, average="micro")
        f1_m.update(f1_value)
        p_value = precision(gt, pre, average="micro", zero_division=0)
        p_m.update(p_value)
        r_value = recall(gt, pre, average="micro")
        r_m.update(r_value)
        acc_value = accuracy(gt, pre)
        acc_m.update(acc_value)

    def train(self):
        timer = Timer()
        writer = SummaryWriter(self.record_path)
        print("*" * 25, " TRAINING BEGINS ", "*" * 25)
        start_epoch = self.start_step // len(self.train_loader) + 1
        for epoch_idx in range(start_epoch, self.num_epoch + 1):
            self.f1_meter.reset()
            self.p_meter.reset()
            self.r_meter.reset()
            self.acc_meter.reset()
            self.loss_meter.reset()
            self.optimizer.step()
            self.scheduler.step()
            train_generator = tqdm(enumerate(self.train_loader, 1), position=0, leave=True)

            for batch_idx, data in train_generator:
                global_step = (epoch_idx - 1) * len(self.train_loader) + batch_idx
                self.model.train()
                id, label, _, mask = data[:4]
                if self.gpu:
                    id, mask, label = self.to_cuda(id, mask, label)
                pre = self.model((id, mask))
                loss = self.loss(pre, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lbl = label.cpu().numpy()
                yp = pre.argmax(1).cpu().numpy()
                self.update_metrics(
                    lbl, yp, self.f1_meter, self.p_meter,
                    self.r_meter, self.acc_meter
                )
                self.loss_meter.update(loss.item())

                if global_step % self.log_steps == 0 and writer is not None:
                    writer.add_scalar("train/f1", self.f1_meter.avg, global_step)
                    writer.add_scalar("train/loss", self.loss_meter.avg, global_step)
                    writer.add_scalar("train/lr", self.scheduler.get_lr()[0], global_step)

                train_generator.set_description(
                    "Train Epoch %d (%d/%d), "
                    "Global Step %d, Loss %.4f, f1 %.4f, p %.4f, r %.4f, acc %.4f, LR %.6f" % (
                        epoch_idx, batch_idx, len(self.train_loader), global_step,
                        self.loss_meter.avg, self.f1_meter.avg,
                        self.p_meter.avg, self.r_meter.avg,
                        self.acc_meter.avg,
                        self.scheduler.get_lr()[0]
                    )
                )

                # validating process
                if global_step % self.val_steps == 0:
                    print()
                    self.validate(epoch_idx, global_step, timer, writer)

                # when num_steps has been set and the training process will
                # be stopped earlier than the specified num_epochs, then stop.
                if self.num_steps is not None and global_step == self.num_steps:
                    if writer is not None:
                        writer.close()
                    print()
                    print("*" * 25, " TRAINING ENDS ", "*" * 25)
                    return

            train_generator.close()
            print()
        writer.close()
        print("*" * 25, " TRAINING ENDS ", "*" * 25)

    def validate(self, epoch, step, timer, writer):
        with torch.no_grad():
            f1_meter = AverageMeter()
            p_meter = AverageMeter()
            r_meter = AverageMeter()
            acc_meter = AverageMeter()
            loss_meter = AverageMeter()
            val_generator = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
            for val_idx, data in val_generator:
                self.model.eval()
                id, label, _, mask = data[:4]
                if self.gpu:
                    id, mask, label = self.to_cuda(id, mask, label)
                pre = self.model((id, mask))
                loss = self.loss(pre, label)

                lbl = label.cpu().numpy()
                yp = pre.argmax(1).cpu().numpy()
                self.update_metrics(lbl, yp, f1_meter, p_meter, r_meter, acc_meter)
                loss_meter.update(loss.item())

                val_generator.set_description(
                    "Eval Epoch %d (%d/%d), Global Step %d, Loss %.4f, "
                    "f1 %.4f, p %.4f, r %.4f, acc %.4f" % (
                        epoch, val_idx, len(self.val_loader), step,
                        loss_meter.avg, f1_meter.avg,
                        p_meter.avg, r_meter.avg, acc_meter.avg
                    )
                )

            print("Eval Epoch %d, f1 %.4f" % (epoch, f1_meter.avg))
            if writer is not None:
                writer.add_scalar("val/loss", loss_meter.avg, step)
                writer.add_scalar("val/f1", f1_meter.avg, step)
                writer.add_scalar("val/precision", p_meter.avg, step)
                writer.add_scalar("val/recall", r_meter.avg, step)
                writer.add_scalar("val/acc", acc_meter.avg, step)
            if f1_meter.avg > self.best_f1:
                self.best_f1 = f1_meter.avg
                self.best_step = step
                self._save_ckpt(step, best=True)
            print("Best Step %d, Best f1 %.4f, Running Time: %s, Estimated Time: %s" % (
                self.best_step, self.best_f1, timer.measure(), timer.measure(step / self.num_steps)
            ))
            self._save_ckpt(step, best=False, f=f1_meter.avg, p=p_meter.avg, r=r_meter.avg)

    def test(self):
        # t_idx = random.randint(0, self.val_bs)
        t_idx = random.randint(0, 5)
        with torch.no_grad():
            self.fixed_randomness()  # for reproduction

            # for writing the total predictions to disk
            data_idxs = list()
            all_preds = list()

            # for ploting P-R Curve
            predicts = list()
            truths = list()

            # for showing predicted samples
            show_ctxs = list()
            pred_lbls = list()
            targets = list()

            f1_meter = AverageMeter()
            p_meter = AverageMeter()
            r_meter = AverageMeter()
            accuracy_meter = AverageMeter()
            test_generator = tqdm(enumerate(self.test_loader, 1))
            for idx, data in test_generator:
                self.model.eval()
                id, label, _, mask, data_idx = data
                if self.gpu:
                    id, mask, label = self.to_cuda(id, mask, label)
                pre = self.model((id, mask))

                lbl = label.cpu().numpy()
                yp = pre.argmax(1).cpu().numpy()
                self.update_metrics(lbl, yp, f1_meter, p_meter, r_meter, accuracy_meter)

                test_generator.set_description(
                    "Test %d/%d, f1 %.4f, p %.4f, r %.4f, acc %.4f"
                    % (idx, len(self.test_loader), f1_meter.avg,
                       p_meter.avg, r_meter.avg, accuracy_meter.avg)
                )

                data_idxs.append(data_idx.numpy())
                all_preds.append(yp)

                predicts.append(torch.select(pre, dim=1, index=1).cpu().numpy())
                truths.append(lbl)

                # show some of the sample
                ctx = torch.select(id, dim=0, index=t_idx).detach()
                ctx = self.model.tokenizer.convert_ids_to_tokens(ctx)
                ctx = "".join([_ for _ in ctx if _ not in [PAD, CLS]])
                yp = yp[t_idx]
                lbl = lbl[t_idx]

                show_ctxs.append(ctx)
                pred_lbls.append(yp)
                targets.append(lbl)

            print("*" * 25, " SAMPLE BEGINS ", "*" * 25)
            for c, t, l in zip(show_ctxs, targets, pred_lbls):
                print("ctx: ", c, " gt: ", t, " est: ", l)
            print("*" * 25, " SAMPLE ENDS ", "*" * 25)
            print("Test, FINAL f1 %.4f, "
                  "p %.4f, r %.4f, acc %.4f\n" %
                  (f1_meter.avg, p_meter.avg, r_meter.avg, accuracy_meter.avg))

            # output the final results to disk
            data_idxs = np.concatenate(data_idxs, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            write_predictions(
                self.val_path, os.path.join(self.record_path, "results.txt"),
                data_idxs, all_preds, delimiter=self.delimiter, skip_first=self.skip_first
            )

            # output the p-r values for future plotting P-R Curve
            predicts = np.concatenate(predicts, axis=0)
            truths = np.concatenate(truths, axis=0)
            values = precision_recall_curve(truths, predicts)
            with open(os.path.join(self.record_path, "pr.values"), "wb") as f:
                pickle.dump(values, f)
            p_value, r_value, _ = values

            # plot P-R Curve if specified
            if arg.image:
                plt.figure()
                plt.plot(
                    p_value, r_value,
                    label="%s (ACC: %.2f, F1: %.2f)"
                          % (self.model_name, accuracy_meter.avg, f1_meter.avg)
                )
                plt.legend(loc="best")
                plt.title("2-Classes P-R curve")
                plt.xlabel("precision")
                plt.ylabel("recall")
                plt.savefig(os.path.join(self.record_path, "P-R.png"))
                plt.show()


if __name__ == '__main__':
    arg = arg_parser()
    pro = arg.proto

    with open(pro) as proto_file:
        proto = yaml.load(proto_file, Loader=yaml.FullLoader)

    print("==> Stage: %s begins..." % arg.stage)
    if not torch.cuda.is_available():
        print("==> You are currently using CPU-only device,"
              " please wait patiently for preparing data and loading the model...")
    if arg.stage == "train":
        trainer = Trainer(proto, stage="train")
        trainer.train()
    elif arg.stage == "test":
        trainer = Trainer(proto, stage="test")
        trainer.test()
    else:
        raise ValueError("The argument 'stage' should only be either 'train' or 'test', "
                         "but found %s." % arg.stage)
    print("\n==> Process ends normally.")
