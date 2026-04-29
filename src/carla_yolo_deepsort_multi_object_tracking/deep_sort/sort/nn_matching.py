# vim: expandtab:ts=4:sw=4
import numpy as np

class NearestNeighborMatching(object):
    """
    This call has methods that perform nearest neighbour distance between 
    targets features and samples features.
 
    """

    def __init__(self, metric, matching_threshold, budget=None):

        """
        Parameters
        ----------
            metric : str (eculidean or cosine)
            matching_threshold: float , threshold value 
            budget : Optional[int]
        """

        if metric == "euclidean":
            self.metric_type = nn_euclidean
        elif metric == "cosine":
            self.metric_type = nn_cosine
        else:
            raise ValueError(
                "Invalid metric type; must be either 'euclidean' or 'cosine'")

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):

        """Update the distance metric with new data.

        Parameters
        ----------
            features : ndarray , shape (N,M)
            targets : ndarray , 
            active_targets : List[int]

        """
        for feature,target in zip(features,targets):
            self.samples.setdefault(target,[]).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}        
        
    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray , shape (N,M)
        targets : List[int]

        """
        cost_matrix = np.zeros((len(targets), len(features)))

        for i in range(len(targets)):
            target=targets[i]
            target_sample = self.samples[target]
            cost_matrix[i, :] = self.metric_type(target_sample, features)

        return cost_matrix

def pair_dist(a, b):
    """
        This function calculate the squared distance 
        between  pairs of points in `a` and `b`.

    """
    a, b = np.asarray(a), np.asarray(b)

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    square_a = np.square(a).sum(axis=1)
    square_b = np.square(b).sum(axis=1)

    r2 = -2. * np.dot(a, b.T) + square_a[:, None] + square_b[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2


def cos_dist(a, b, data_is_normalized=False):

    """
        This function calculate the cosine distance 
        between  pairs of points in `a` and `b`

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)

    return 1. - np.dot(a, b.T)
def nn_euclidean(x, y):

    distances = pair_dist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def nn_cosine(x, y):

    distances = cos_dist(x, y)
    return distances.min(axis=0)