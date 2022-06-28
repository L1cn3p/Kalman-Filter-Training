from numpy import dot, array, diag, eye, zeros, sqrt, square
import matplotlib
from numpy.linalg import inv, det
from matplotlib import pyplot as plt

# true trajectory of the vehicle
def div_fun(x):
    if x >= -400 and x < 0:
        return 300
    elif x >= 0 and x <= 300:
        return sqrt(90000 - square(x))
    else:
        return 0

class KalmanFilter:
    def __init__(self):
        self.transition_matrix = None
        self.initial_position = (x, y)
        self._current_position = None

        
    
    @property
    def current_position(self):
        # return the current position of the object
        return self._current_position
    
    @current_position.setter
    def current_position(self, x, y):
        self._current_position = (x, y)
        self.update()

    # the update method should update the transition matrix based on the predicted position and the actual position
    def update(self):
        pass

    # the predict method should accept a vector (actual) and return a predict positon
    def predict_state(self):
        # pred state = trans matrix * estimated state at previous step + control mat * control var + process noise
        pass



if __name__ == '__main__':
    # time step of mobile movement
    dt = 1
    # Initialization of state matrices
    # X is a initial system state vector at time step 0
    X = array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    # P is the initial uncertainty(covariance) matrix of the current state
    P = diag((500, 500, 500, 500, 500, 500))
    # F is the state transition matrix
    F = array([[1, dt, 0.5, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0.5], [0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 1]])

    sigma_a = 0.2
    # Q is the process noise matrix
    Q = array([[1/4, 1/2, 1/2, 0, 0, 0], [1/2, 1, 1, 0, 0, 0], [1/2, 1, 1, 0, 0, 0],
               [0, 0, 0, 1/4, 1/2, 1/2], [0, 0, 0, 1/2, 1, 1],
               [0, 0, 0,1/2, 1, 1]])*square(sigma_a)

    # G is the control matrix (not used in this example)
    G = eye(X.shape[0])
    # U is a control variable (not used in this example)
    U = zeros((X.shape[0], 1))
    # H is the observation matrix
    H = array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    # R is the Measurement Uncertainly
    sigma_xm = 3
    sigma_ym = 3
    R = array([[square(sigma_xm), 0], [0, square(sigma_ym)]])

    # measurement trajectory of the vehicle
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
    # z1 = (x(1),y(1)) = (-393.66,300.4),z2 = (x(2),y(2)) = (-375.93,301.78)
    # Number of iterations in Kalman Filter
    N_iter = len(x)
    y_true = []
    for i in enumerate(x):
        y_true.append(div_fun(i[1]))
    plt.plot(x, y_true)
    plt.show()
