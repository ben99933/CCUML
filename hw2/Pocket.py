import numpy as np
import argparse
from DataLoader import DataLoader
import os
import matplotlib.pyplot as plt
import time


def pocket(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    weight_matrix = np.zeros(3)
    s = time.time()
    ############ START ##########
    x = np.array(DataLoader.data)  # x = [1, x1, x2]
    label = np.array(DataLoader.label) # label = [1 or -1]
    # shuffle the data
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    label = label[idx]

    MAX_ITERATION = 50000
    n = len(x)
    dim = len(x[0])
    iteration: int = 0
    best_accuracy: float = 0
    best_weight: np.ndarray = weight_matrix.copy()
    print("iteration, accuracy, best_accuracy")
    while True:
        iteration += 1
        error_count: int = 0
        wrong:list = []
        for i in range(n):
            dot = np.dot(weight_matrix, x[i])
            sign = np.sign(dot)
            sign = -1 if sign == 0 else sign
            if sign != label[i]:
                wrong.append(i)
        if not wrong:
            break
        pick = np.random.choice(wrong)
        weight_matrix += np.multiply(label[pick], x[pick])
        accuracy = np.sum(np.sign(np.dot(x, weight_matrix)) == label) / n
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = weight_matrix.copy()
        # print("%d, %f, %f" % (iteration, accuracy, best_accuracy))
        if iteration >= MAX_ITERATION:
            break
    weight_matrix = best_weight.copy()
    print("accuracy = %f" % (best_accuracy))
    print("iteration = %d" % (iteration))
    ############ END ############
    e = time.time()
    print("ex time = %f" % (e-s))
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = pocket(DataLoader=Loader)

    # This part is for plotting the graph
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(Loader.cor_x_pos, Loader.cor_y_pos,
                c='b', label='pos data')
    plt.scatter(Loader.cor_x_neg, Loader.cor_y_neg,
                c='r', label='neg data')

    x = np.linspace(-1000, 1000, 100)
    # This is the base line
    y1 = 3*x+5
    # This is your split line
    y2 = (updated_weight[1]*x + updated_weight[0]) / (-updated_weight[2])
    plt.plot(x, y1, 'g', label='base line', linewidth='1')
    plt.plot(x, y2, 'y', label='split line', linewidth='1')
    plt.legend()
    plt.show()
    # plt.savefig(f"./output/{args.path}.png")

if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
