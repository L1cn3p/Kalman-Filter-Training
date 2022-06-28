from numpy import dot, array, diag, eye, zeros, sqrt, square
import matplotlib
from numpy.linalg import inv, det
from matplotlib import pyplot as plt


def div_fun(x):
    if x >= -400 and x < 0:
        return 300
    elif x >= 0 and x <= 300:
        return sqrt(90000 - square(x))
    else:
        return 0


if __name__ == '__main__':
    # time step of mobile movement
    dt = 1
    # Initialization of state matrices
    X = array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    P = diag((500, 500, 500, 500, 500, 500))
    A = array([[1, dt, 0.5, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0.5], [0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 1]])
    Q = array([[0.01, 0.02, 0.02, 0, 0, 0], [0.02, 0.04, 0.04, 0, 0, 0], [0.02, 0.04, 0.04, 0, 0, 0],
               [0, 0, 0, 0.01, 0.02, 0.02], [0, 0, 0, 0.02, 0.04, 0.04],
               [0, 0, 0, 0.02, 0.04, 0.04]])
    B = eye(X.shape[0])
    U = zeros((X.shape[0], 1))
    H = array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    R = array([[9, 0], [0, 9]])

    # measurement values
    x = (-393.66, -375.93, -351.04, -328.96, -299.35,
         -273.36, -245.89, -222.58, -198.03, -174.17,
         -146.32, -123.72, -103.47, -78.23, -52.63, -23.34,
         25.96, 49.72, 76.94, 95.38, 119.83, 144.01, 161.84, 180.56,
         201.42, 222.62, 239.4, 252.51, 266.26, 271.75, 277.4,
         294.12, 301.23, 291.8, 299.89)
    y = (300.4, 301.78, 295.1, 305.19, 301.06, 302.05,
         300, 303.57, 296.33, 297.65, 297.41, 299.61,
         299.6, 302.39, 295.04, 300.09, 294.72, 298.61,
         294.64, 284.88, 272.82, 264.93, 251.46, 241.27,
         222.98, 203.73, 184.1, 166.12, 138.71, 119.71,
         100.41, 79.76, 50.62, 32.99, 2.14)
    # Number of iterations in Kalman Filter
    N_iter = len(x)
    y_true = []
    for i in enumerate(x):
        y_true.append(div_fun(i[1]))
    plt.plot(x, y_true)
    plt.show()
