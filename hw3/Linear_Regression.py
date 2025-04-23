import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt


def Linear_Regression(DataLoader: DataLoader):
    """
    Do the Linear_Regression here on your own.
    weights -> 2 * 1 resulted weight matrix  
    Ein -> The in-sample error
    """
    weights = np.zeros(2)
    Ein = 0
    ############ START ##########
    data = np.array(DataLoader.data)

    X = data[:, 0]
    Y = data[:, 1]
    n = len(X)
    # y = Ax
    A = np.vstack([np.ones(n), X]).T
    # w = (A^T A)^-1 A^T y
    weights = np.linalg.inv(A.T @ A) @ A.T @ Y

    y_pred = A @ weights
    errors = Y - y_pred
    Ein = np.mean(errors**2)
    grad_Ein = (2 / n) * (((A.T @ A) @ weights) - (A.T @ Y))

    print(f'weights[0]:{weights[0]}, weights[1]:{weights[1]}')
    print(f'Ein: {Ein}')
    print(f'grad_Ein: {grad_Ein}')
    ############ END ############
    return weights, Ein


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    weights, Ein = Linear_Regression(DataLoader=Loader)

    # This part is for plotting the graph
    plt.title(
        'Linear Regression, Ein = %.2f' % (Ein))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    Data = np.array(Loader.data)
    plt.scatter(Data[:, 0], Data[:, 1], c='b', label='data')

    x = np.linspace(-100, 100, 10)
    # This is your regression line
    y = weights[1]*x + weights[0]
    plt.plot(x, y, 'g', label='regression line', linewidth='1')
    plt.legend()
    plt.show()
    # plt.savefig("Linear_Regression.png")

if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
