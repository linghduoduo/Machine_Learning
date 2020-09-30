import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs
np.random.seed(123)


class KMeans():
    def __init__(self, n_clusters=4):
        self.k = n_clusters

    def fit(self, data):
        """
        Fits the k-means model to the given dataset
        """
        n_samples, _ = data.shape
        # initialize cluster centers
        self.centers = np.array(random.sample(list(data), self.k))
        self.initial_centers = np.copy(self.centers)

        # We will keep track of whether the assignment of data points
        # to the clusters has changed. If it stops changing, we are
        # done fitting the model
        old_assigns = None
        n_iters = 0

        while True:
            new_assigns = [self.classify(datapoint) for datapoint in data]

            if new_assigns == old_assigns:
                print(f"Training finished after {n_iters} iterations!")
                return

            old_assigns = new_assigns
            n_iters += 1

            # recalculate centers
            for id_ in range(self.k):
                points_idx = np.where(np.array(new_assigns) == id_)
                datapoints = data[points_idx]
                self.centers[id_] = datapoints.mean(axis=0)

    def l2_distance(self, datapoint):
        dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis=1))
        return dists

    def classify(self, datapoint):
        """
        Given a datapoint, compute the cluster closest to the
        datapoint. Return the cluster ID of that cluster.
        """
        dists = self.l2_distance(datapoint)
        return np.argmin(dists)

    def plot_clusters(self, data):
        plt.figure(figsize=(12, 10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:, 0], data[:, 1], marker='.', c=y)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='r')
        plt.scatter(self.initial_centers[:, 0],
                    self.initial_centers[:, 1], c='k')
        plt.show()


# Simulate the data
X, y = make_blobs(centers=4, n_samples=1000)
print(f'Shape of dataset: {X.shape}')

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset with 4 clusters")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()


# Initializing and fitting the model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Plot initial and final cluster centers
kmeans.plot_clusters(X)

# Another application
# Initialisation

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# df = pd.DataFrame({
#     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })


# np.random.seed(200)
# k = 3
# # centroids[i] = [x, y]
# centroids = {
#     i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
#     for i in range(k)
# }

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color='k')
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()


# ## Assignment Stage
# ## Randomly pick k centroids fromt eh sample points as initial cluster centers.
# ## Aassign each sample to the nearest centroid.

# def assignment(df, centroids):
#     for i in centroids.keys():
#         # sqrt((x1 - x2)^2 - (y1 - y2)^2)
#         df['distance_from_{}'.format(i)] = (
#             np.sqrt(
#                 (df['x'] - centroids[i][0]) ** 2
#                 + (df['y'] - centroids[i][1]) ** 2
#             )
#         )
#     centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
#     df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
#     df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
#     df['color'] = df['closest'].map(lambda x: colmap[x])
#     return df

# df = assignment(df, centroids)
# print(df.head())

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()


# ## Update Stage
# ## UPdate centroid

# import copy

# old_centroids = copy.deepcopy(centroids)

# def update(k):
#     for i in centroids.keys():
#         centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
#         centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
#     return k

# centroids = update(centroids)

# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# for i in old_centroids.keys():
#     old_x = old_centroids[i][0]
#     old_y = old_centroids[i][1]
#     dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
#     dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
#     ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
# plt.show()


# ## Repeat Assigment Stage

# df = assignment(df, centroids)

# # Plot results
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()


# # Continue until all assigned categories don't change any more
# while True:
#     closest_centroids = df['closest'].copy(deep=True)
#     centroids = update(centroids)
#     df = assignment(df, centroids)
#     if closest_centroids.equals(df['closest']):
#         break

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()


"""
Requirements:
  - sklearn
  - numpy
  - matplotlib
Python:
  - 3.5
Inputs:
  - X , a 2D numpy array of features.
  - k , number of clusters to create.
  - initial_centroids , initial centroid values generated by utility function(mentioned
    in usage).
  - maxiter , maximum number of iterations to process.
  - heterogeneity , empty list that will be filled with hetrogeneity values if passed
    to kmeans func.
Usage:
  1. define 'k' value, 'X' features array and 'hetrogeneity' empty list
  2. create initial_centroids,
        initial_centroids = get_initial_centroids(
            X,
            k,
            seed=0 # seed value for initial centroid generation,
                   # None for randomness(default=None)
            )
  3. find centroids and clusters using kmeans function.
        centroids, cluster_assignment = kmeans(
            X,
            k,
            initial_centroids,
            maxiter=400,
            record_heterogeneity=heterogeneity,
            verbose=True # whether to print logs in console or not.(default=False)
            )
  4. Plot the loss function, hetrogeneity values for every iteration saved in
     hetrogeneity list.
        plot_heterogeneity(
            heterogeneity,
            k
        )
  5. Transfers Dataframe into excel format it must have feature called
      'Clust' with k means clustering numbers in it.
"""


### Another Implementation
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

TAG = "K-MEANS-CLUST/ "


def get_initial_centroids(data, k, seed=None):
    """Randomly choose k data points as initial centroids"""
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0]  # number of data points

    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices, :]

    return centroids


def assign_clusters(data, centroids):

    # Compute distances between each data point and the set of centroids:
    # Fill in the blank (RHS only)
    distances_from_centroids = pairwise_distances(X, centroids, metric="euclidean")

    # Compute cluster assignments for each data point:
    # Fill in the blank (RHS only)
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)

    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment == i]
        # Compute the mean of the data points. Fill in the blank (RHS only)
        centroid = member_data_points.mean(axis=0)
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)

    return new_centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):

    heterogeneity = 0.0
    for i in range(k):

        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment == i, :]

        if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
            # Compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric="euclidean")
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7, 4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel("# Iterations")
    plt.ylabel("Heterogeneity")
    plt.title(f"Heterogeneity of clustering over time, K={k:d}")
    plt.rcParams.update({"font.size": 16})
    plt.show()


def kmeans(
    data, k, initial_centroids, maxiter=500, record_heterogeneity=None, verbose=False
):
    """This function runs k-means on given data and initial set of centroids.
    maxiter: maximum number of iterations to run.(default=500)
    record_heterogeneity: (optional) a list, to store the history of heterogeneity
                          as function of iterations
                          if None, do not store the history.
    verbose: if True, print how many data points changed their cluster labels in
                          each iteration"""
    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(maxiter):
        if verbose:
            print(itr, end="")

        # 1. Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)

        # 2. Compute a new centroid for each of the k clusters, averaging all data
        #    points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)

        # Check for convergence: if none of the assignments changed, stop
        if (
            prev_cluster_assignment is not None
            and (prev_cluster_assignment == cluster_assignment).all()
        ):
            break

        # Print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            if verbose:
                print(
                    "    {:5d} elements changed their cluster assignment.".format(
                        num_changed
                    )
                )

        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            # YOUR CODE HERE
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


# Mock test below
if False:  # change to true to run this test case.
    from sklearn import datasets as ds

    dataset = ds.load_iris()
    k = 3
    heterogeneity = []
    initial_centroids = get_initial_centroids(dataset["data"], k, seed=0)
    centroids, cluster_assignment = kmeans(
        dataset["data"],
        k,
        initial_centroids,
        maxiter=400,
        record_heterogeneity=heterogeneity,
        verbose=True,
    )
    plot_heterogeneity(heterogeneity, k)


def ReportGenerator(
    df: pd.DataFrame, ClusteringVariables: np.array, FillMissingReport=None
) -> pd.DataFrame:
    """
    Function generates easy-erading clustering report. It takes 2 arguments as an input:
        DataFrame - dataframe with predicted cluester column;
        FillMissingReport - dictionary of rules how we are going to fill missing
        values of for final report generate (not included in modeling);
    in order to run the function following libraries must be imported:
        import pandas as pd
        import numpy as np
    >>> data = pd.DataFrame()
    >>> data['numbers'] = [1, 2, 3]
    >>> data['col1'] = [0.5, 2.5, 4.5]
    >>> data['col2'] = [100, 200, 300]
    >>> data['col3'] = [10, 20, 30]
    >>> data['Cluster'] = [1, 1, 2]
    >>> ReportGenerator(data, ['col1', 'col2'], 0)
               Features               Type   Mark           1           2
    0    # of Customers        ClusterSize  False    2.000000    1.000000
    1    % of Customers  ClusterProportion  False    0.666667    0.333333
    2              col1    mean_with_zeros   True    1.500000    4.500000
    3              col2    mean_with_zeros   True  150.000000  300.000000
    4           numbers    mean_with_zeros  False    1.500000    3.000000
    ..              ...                ...    ...         ...         ...
    99            dummy                 5%  False    1.000000    1.000000
    100           dummy                95%  False    1.000000    1.000000
    101           dummy              stdev  False    0.000000         NaN
    102           dummy               mode  False    1.000000    1.000000
    103           dummy             median  False    1.000000    1.000000
    <BLANKLINE>
    [104 rows x 5 columns]
    """
    # Fill missing values with given rules
    if FillMissingReport:
        df.fillna(value=FillMissingReport, inplace=True)
    df["dummy"] = 1
    numeric_cols = df.select_dtypes(np.number).columns
    report = (
        df.groupby(["Cluster"])[  # construct report dataframe
            numeric_cols
        ]  # group by cluster number
        .agg(
            [
                ("sum", np.sum),
                ("mean_with_zeros", lambda x: np.mean(np.nan_to_num(x))),
                ("mean_without_zeros", lambda x: x.replace(0, np.NaN).mean()),
                (
                    "mean_25-75",
                    lambda x: np.mean(
                        np.nan_to_num(
                            sorted(x)[
                                round((len(x) * 25 / 100)) : round(len(x) * 75 / 100)
                            ]
                        )
                    ),
                ),
                ("mean_with_na", np.mean),
                ("min", lambda x: x.min()),
                ("5%", lambda x: x.quantile(0.05)),
                ("25%", lambda x: x.quantile(0.25)),
                ("50%", lambda x: x.quantile(0.50)),
                ("75%", lambda x: x.quantile(0.75)),
                ("95%", lambda x: x.quantile(0.95)),
                ("max", lambda x: x.max()),
                ("count", lambda x: x.count()),
                ("stdev", lambda x: x.std()),
                ("mode", lambda x: x.mode()[0]),
                ("median", lambda x: x.median()),
                ("# > 0", lambda x: (x > 0).sum()),
            ]
        )
        .T.reset_index()
        .rename(index=str, columns={"level_0": "Features", "level_1": "Type"})
    )  # rename columns
    # calculate the size of cluster(count of clientID's)
    clustersize = report[
        (report["Features"] == "dummy") & (report["Type"] == "count")
    ].copy()  # avoid SettingWithCopyWarning
    clustersize.Type = (
        "ClusterSize"  # rename created cluster df to match report column names
    )
    clustersize.Features = "# of Customers"
    clusterproportion = pd.DataFrame(
        clustersize.iloc[:, 2:].values
        / clustersize.iloc[:, 2:].values.sum()  # calculating the proportion of cluster
    )
    clusterproportion[
        "Type"
    ] = "% of Customers"  # rename created cluster df to match report column names
    clusterproportion["Features"] = "ClusterProportion"
    cols = clusterproportion.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    clusterproportion = clusterproportion[cols]  # rearrange columns to match report
    clusterproportion.columns = report.columns
    a = pd.DataFrame(
        abs(
            report[report["Type"] == "count"].iloc[:, 2:].values
            - clustersize.iloc[:, 2:].values
        )
    )  # generating df with count of nan values
    a["Features"] = 0
    a["Type"] = "# of nan"
    a.Features = report[
        report["Type"] == "count"
    ].Features.tolist()  # filling values in order to match report
    cols = a.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    a = a[cols]  # rearrange columns to match report
    a.columns = report.columns  # rename columns to match report
    report = report.drop(
        report[report.Type == "count"].index
    )  # drop count values except cluster size
    report = pd.concat(
        [report, a, clustersize, clusterproportion], axis=0
    )  # concat report with clustert size and nan values
    report["Mark"] = report["Features"].isin(ClusteringVariables)
    cols = report.columns.tolist()
    cols = cols[0:2] + cols[-1:] + cols[2:-1]
    report = report[cols]
    sorter1 = {
        "ClusterSize": 9,
        "ClusterProportion": 8,
        "mean_with_zeros": 7,
        "mean_with_na": 6,
        "max": 5,
        "50%": 4,
        "min": 3,
        "25%": 2,
        "75%": 1,
        "# of nan": 0,
        "# > 0": -1,
        "sum_with_na": -2,
    }
    report = (
        report.assign(
            Sorter1=lambda x: x.Type.map(sorter1),
            Sorter2=lambda x: list(reversed(range(len(x)))),
        )
        .sort_values(["Sorter1", "Mark", "Sorter2"], ascending=False)
        .drop(["Sorter1", "Sorter2"], axis=1)
    )
    report.columns.name = ""
    report = report.reset_index()
    report.drop(columns=["index"], inplace=True)
    return report


if __name__ == "__main__":
    import doctest

    doctest.testmod()