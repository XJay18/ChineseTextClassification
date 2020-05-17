import numpy as np

np.random.seed(666)

if __name__ == '__main__':
    to_split_path = "./path/to/data"
    train_path = "./path/to/data"
    val_path = "./path/to/data"
    positive = []
    negative = []
    print("Reading data...")
    with open(to_split_path, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            data_idx, ctx, label = line.split("\t")
            if int(label) == 0:
                positive.append(line)
            else:
                negative.append(line)
            line = f.readline()
    print("label == 0: ", len(positive))
    print("label == 1: ", len(negative))
    stop_positive = len(positive) - 2000
    stop_negative = len(negative) - 2000

    np.random.shuffle(positive)
    np.random.shuffle(negative)

    print("Writing data...")
    with open(train_path, "w", encoding="utf-8") as f:
        for _ in range(stop_positive):
            f.writelines(positive[_])
        for _ in range(stop_negative):
            f.writelines(negative[_])
    with open(val_path, "w", encoding="utf-8") as f:
        for _ in range(stop_positive, len(positive)):
            f.writelines(positive[_])
        for _ in range(stop_negative, len(negative)):
            f.writelines(negative[_])
    print("Done")
