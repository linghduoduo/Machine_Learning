###########################################################
# Training with normal equation
###########################################################
import numpy as np

n = 100
X = 2 * np.random.rand(n, 1)
y = 4 + 3 * X + np.random.randn(n, 1)

X_b = np.c_[np.ones((n, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_

## X_train
# To compute the parameters using the normal equation, we add a bias value
# of 1 to each input example
X_b_train = np.c_[np.ones((n_samples)), X_train]
X_b_test = np.c_[np.ones((n_samples_test)), X_test]

reg_normal = LinearRegression()
w_trained = reg_normal.train_normal_equation(X_b_train, y_train)

# Testing normal equation
y_p_train = reg_normal.predict(X_b_train)
y_p_test = reg_normal.predict(X_b_test)

error_train = (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)
print(">>> Error for Normal equation")
print(f"Error on training set: {np.round(error_train, 4)}")
print(f"Error on test set: {np.round(error_test, 4)}")


###########################################################
# Linear regression using batch gradient descent
###########################################################
eta = 0.1  # learning rate
n_iterations = 1000

theta = np.random.randn(2,1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2/n * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

    
###########################################################
# Stochastic Gradient Descent    
###########################################################
m = len(X_b)
np.random.seed(42)
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients              

        
###########################################################
# Mini-batch gradient descent
###########################################################
n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

###########################################################
## LR Gradient Descent Class Implementations
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(123)

class LinearRegression:

    def __init__(self):
        pass

    def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100):
        """
        Trains a linear regression model using gradient descent
        """
        # Step 0: Initialize the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # Step 1: Compute a linear combination of the input features and
            # weights
            y_predict = np.dot(X, self.weights) + self.bias

            # Step 2: Compute cost over training set
            cost = (1 / n_samples) * np.sum((y_predict - y)**2)
            costs.append(cost)

            if i % 100 == 0:
                print(f"Cost at iteration {i}: {cost}")

            # Step 3: Compute the gradients
            dJ_dw = (2 / n_samples) * np.dot(X.T, (y_predict - y))
            dJ_db = (2 / n_samples) * np.sum((y_predict - y))

            # Step 4: Update the parameters
            self.weights = self.weights - learning_rate * dJ_dw
            self.bias = self.bias - learning_rate * dJ_db

        return self.weights, self.bias, costs

    def train_normal_equation(self, X, y):
        """
        Trains a linear regression model using the normal equation
        """
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.bias = 0

        return self.weights, self.bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# We will use a simple training set
X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.randn(500, 1)

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

n_samples, _ = X_train.shape
n_samples_test, _ = X_test.shape

# Training with gradient descent
regressor = LinearRegression()
w_trained, b_trained, costs = regressor.train_gradient_descent(
    X_train, y_train, learning_rate=0.005, n_iters=600)

# Testing gradient descent
n_samples, _ = X_train.shape
n_samples_test, _ = X_test.shape

y_p_train = regressor.predict(X_train)
y_p_test = regressor.predict(X_test)

error_train = (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)
print(">>> Error for gradient descent")
print(f"Error on training set: {np.round(error_train, 4)}")
print(f"Error on test set: {np.round(error_test)}")


"""
Implementation of gradient descent algorithm for minimizing cost of a linear hypothesis function.
"""
import numpy

# List of input, output pairs
train_data = (
    ((5, 2, 3), 15),
    ((6, 5, 9), 25),
    ((11, 12, 13), 41),
    ((1, 1, 1), 8),
    ((11, 12, 13), 41),
)
test_data = (((515, 22, 13), 555), ((61, 35, 49), 150))
parameter_vector = [2, 4, 1, 5]
m = len(train_data)
LEARNING_RATE = 0.009

def _error(example_no, data_set="train"):
    """
    :param data_set: train data or test data
    :param example_no: example number whose error has to be checked
    :return: error in example pointed by example number.
    """
    return calculate_hypothesis_value(example_no, data_set) - output(
        example_no, data_set
    )


def _hypothesis_value(data_input_tuple):
    """
    Calculates hypothesis function value for a given input
    :param data_input_tuple: Input tuple of a particular example
    :return: Value of hypothesis function at that point.
    Note that there is an 'biased input' whose value is fixed as 1.
    It is not explicitly mentioned in input data.. But, ML hypothesis functions use it.
    So, we have to take care of it separately. Line 36 takes care of it.
    """
    hyp_val = 0
    for i in range(len(parameter_vector) - 1):
        hyp_val += data_input_tuple[i] * parameter_vector[i + 1]
    hyp_val += parameter_vector[0]
    return hyp_val


def output(example_no, data_set):
    """
    :param data_set: test data or train data
    :param example_no: example whose output is to be fetched
    :return: output for that example
    """
    if data_set == "train":
        return train_data[example_no][1]
    elif data_set == "test":
        return test_data[example_no][1]


def calculate_hypothesis_value(example_no, data_set):
    """
    Calculates hypothesis value for a given example
    :param data_set: test data or train_data
    :param example_no: example whose hypothesis value is to be calculated
    :return: hypothesis value for that example
    """
    if data_set == "train":
        return _hypothesis_value(train_data[example_no][0])
    elif data_set == "test":
        return _hypothesis_value(test_data[example_no][0])


def summation_of_cost_derivative(index, end=m):
    """
    Calculates the sum of cost function derivative
    :param index: index wrt derivative is being calculated
    :param end: value where summation ends, default is m, number of examples
    :return: Returns the summation of cost derivative
    Note: If index is -1, this means we are calculating summation wrt to biased
        parameter.
    """
    summation_value = 0
    for i in range(end):
        if index == -1:
            summation_value += _error(i)
        else:
            summation_value += _error(i) * train_data[i][0][index]
    return summation_value


def get_cost_derivative(index):
    """
    :param index: index of the parameter vector wrt to derivative is to be calculated
    :return: derivative wrt to that index
    Note: If index is -1, this means we are calculating summation wrt to biased
        parameter.
    """
    cost_derivative_value = summation_of_cost_derivative(index, m) / m
    return cost_derivative_value


def run_gradient_descent():
    global parameter_vector
    # Tune these values to set a tolerance value for predicted output
    absolute_error_limit = 0.000002
    relative_error_limit = 0
    j = 0
    while True:
        j += 1
        temp_parameter_vector = [0, 0, 0, 0]
        for i in range(0, len(parameter_vector)):
            cost_derivative = get_cost_derivative(i - 1)
            temp_parameter_vector[i] = (
                parameter_vector[i] - LEARNING_RATE * cost_derivative
            )
        if numpy.allclose(
            parameter_vector,
            temp_parameter_vector,
            atol=absolute_error_limit,
            rtol=relative_error_limit,
        ):
            break
        parameter_vector = temp_parameter_vector
    print(("Number of iterations:", j))


def test_gradient_descent():
    for i in range(len(test_data)):
        print(("Actual output value:", output(i, "test")))
        print(("Hypothesis output:", calculate_hypothesis_value(i, "test")))


if __name__ == "__main__":
    run_gradient_descent()
    print("\nTesting gradient descent for a linear hypothesis function.\n")
    test_gradient_descent()
