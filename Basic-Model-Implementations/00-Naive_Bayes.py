###########################################################
## Bernoulli Naive Bayes Example
###########################################################
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# The features in X are broken down as follows:
# [Walks like a duck, Talks like a duck, Is small]
#
# Walks like a duck: 0 = False, 1 = True
# Talks like a duck: 0 = False, 1 = True
# Is small: 0 = False, 1 = True

# Some data is created to train with
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
# These are our target values (Classes: Duck or Not a duck)
y = np.array(['Duck', 'Not a Duck', 'Not a Duck'])

# This is the code we need for the Bernoulli model
clf = BernoulliNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents things that are and aren't ducks.\n")
print("We have trained a Bernoulli model on our data set.\n")
print(("Let's consider a new input that:\n"
       "   Walks like a duck\n"
       "   Talks like a duck\n"
       "   Is large\n"))
print("What does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 1, 1]])[0])


###########################################################
## Gaussian Naive Bayes Example
###########################################################
import numpy as np
from sklearn.naive_bayes import GaussianNB

# The features in X are broken down as follows:
# [Red %, Green %, Blue %]

# Some data is created to train with
X = np.array([[.5, 0, .5], [1, 1, 0], [0, 0, 0]])
# These are our target values (Classes: Purple, Yellow, or Black)
y = np.array(['Purple', 'Yellow', 'Black'])

# This is the code we need for the Gaussian model
clf = GaussianNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents RGB triples and their associated colors.\n")
print("We have trained a Gaussian model on our data set.\n")
print("Let's consider a new input with 100% red, 0% green, and 100% blue.\n")
print("What color does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 0, 1]])[0])


## Another example of Gaussian Bayes
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():

    """
    Gaussian Naive Bayes Example using sklearn function.
    Iris type dataset is used to demonstrate algorithm.
    """

    # Load Iris dataset
    iris = load_iris()

    # Split dataset into train and test data
    X = iris["data"]  # features
    Y = iris["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1
    )

    # Gaussian Naive Bayes
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)

    # Display Confusion Matrix
    plot_confusion_matrix(
        NB_model,
        x_test,
        y_test,
        display_labels=iris["target_names"],
        cmap="Blues",
        normalize="true",
    )
    plt.title("Normalized Confusion Matrix - IRIS Dataset")
    plt.show()
    if __name__ == "__main__":
    main()


###########################################################
## Multinomial Naive Bayes Example
###########################################################

import numpy as np
from sklearn.naive_bayes import MultinomialNB

# The features in X are broken down as follows:
# [Size, Weight, Color]
#
# Size: 0 = Small, 1 = Moderate, 2 = Large
# Weight: 0 = Light, 1 = Moderate, 2 = Heavy
# Color: 0 = Red, 1 = Blue, 2 = Brown

# Some data is created to train with
X = np.array([[1, 1, 0], [0, 0, 1], [2, 2, 2]])
# These are our target values (Classes: Apple, Blueberry, or Coconut)
y = np.array(['Apple', 'Blueberry', 'Coconut'])

# This is the code we need for the Multinomial model
clf = MultinomialNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents fruits and their characteristics.\n")
print("We have trained a Multinomial model on our data set.\n")
print("Let's consider a new input that is moderately sized, heavy, and red.\n")
print("What fruit does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 2, 0]])[0])    


