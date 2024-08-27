from .EVM import EVM

import scipy.spatial
import sklearn.cluster
import numpy
import multiprocessing
import random
import h5py

import logging
logger = logging.getLogger("evm")

def _cluster(args):
  i,kmeans = args
  kmeans.fit(_training_data[i])
  return kmeans.cluster_centers_

def _train(args):
  self,i = args
  global _training_data
  positives = _training_data[i]
  negatives = []
  if _cluster_centers is None:
    # collect negatives randomly from training data
    for j in range(len(_training_data)):
      if i != j:
        if self.number_of_negatives is None:
          negatives.extend(_training_data[j])
        else:
          negatives.extend(random.sample(_training_data[j], min(len(_training_data[j]), self.number_of_negatives)))
  else:
    # first, compute the distances to all cluster centers
    distances = [[[self.distance_function(p, c)
                  for p in positives]
                  for c in center]
                  for j,center in enumerate(_cluster_centers) if i != j]
    # sort them, use minimum distance
    min_distances = sorted(((numpy.min(d), j) for j,d in enumerate(distances)), key=lambda x: x[0])

    # second, extract all negatives from the closest classes
    neg_count = self.number_of_negatives if self.number_of_negatives is not None else len(min_distances)
    for _,j in min_distances[:neg_count]:
      negatives.extend(_training_data[j + (j > i)])

  # now, train the EVM
  evm = EVM(self.tailsize, self.cover_threshold, self.distance_multiplier, self.distance_function, self.include_cover_probability, log_level='debug')
  evm.train(positives, negatives)

  return evm, i

def _probabilities(evm, points):
  return evm.probabilities(points)

def _probabilities_parallel(e):
  global _evms, _points
  return _probabilities(_evms[e], _points)


class MultipleEVM (object):
  """Computes a list of EVMs from a larger set of features

  For a given list of features from different classes, this class computes an EVM for each of the classes by taking all the other classes as negative training samples.
  There are two different ways implemented on how to select the negative training samples:

  1. negative training examples are randomly sampled from each of the negative classes

  2. negative examples are taken from the classes with the lowest distance to the class in question.
     Therefore, all classes are clustered using K-Means.
     During sampling, the distance to all cluster centers of negative classes is computed, and all features of the closest classes are taken as negatives.

  .. warning:: This class is not thread-safe! Please make sure to train only a single MultiEVM at once.

  Parameters
  ----------

  tailsize : int
    The number of sample distances to include in the Weibull fitting

  cover_threshold : float or ``None``
    If given, the EVMs compute a set cover with the given cover threshold

  cluster_centers : int or None
    If given, a K-Means with the given number of cluster centers will be computed for each class, and the negatives of each class will be taken from the closest classes.
    If ``None`` (the default), negatives will be sampled from all other classes.

  number_of_negatives : int or ``None``
    This parameter depends on ``cluster_centers``.
    If ``cluster_centers`` is ``None``, this parameter provides the (maximum) number of features to sample from each class.
    Otherwise, this parameter defines the number of the closest negative classes, from which all features will be taken.
    If this parameter is ``None`` (the default), no sampling is performed.

  distance_multiplier, distance_function: see :py:class:`EVM`
  """

  def __init__(self,
    tailsize,
    cover_threshold = None,
    cluster_centers = None,
    number_of_negatives = None,
    distance_multiplier = 0.5,
    distance_function = scipy.spatial.distance.cosine,
    include_cover_probability = False
  ):

    if isinstance(tailsize, (str, h5py.Group)):
      return self.load(tailsize)

    self.tailsize = tailsize
    self.cover_threshold = cover_threshold
    self.cluster_centers = cluster_centers
    self.number_of_negatives = number_of_negatives
    self.distance_function = distance_function
    self.distance_multiplier = distance_multiplier
    self.include_cover_probability = include_cover_probability

    self._evms = None


  def _cluster_data(self, data, parallel, **kwargs):
    """Clusters the data using K-Means, and returns the means for each class"""
    global _training_data
    _training_data = data
    arguments = ((i, sklearn.cluster.KMeans(self.cluster_centers, **kwargs)) for i in range(len(data)))
    if parallel is None:
      return [_cluster(a) for a in arguments]
    else:
      pool = multiprocessing.Pool(parallel)
      result = pool.map(_cluster, arguments)
      pool.close()
      pool.join()
      return result


  def _train_evms(self, data, cluster_centers, parallel=None, **kwargs):
    """Trains the EVMs"""
    global _training_data, _cluster_centers
    _training_data = data
    _cluster_centers = cluster_centers
    models = []
    arguments = ((self, i) for i in range(len(data)))
    if parallel is None:
      for arg in arguments:
        models.append(_train(arg)[0])
    else:
      pool = multiprocessing.Pool(parallel)
      models = [None]*len(data)
      for evm, j in pool.imap_unordered(_train, arguments):
        models[j] = evm
      pool.close()
      pool.join()
    return models


  def train(self, data, parallel=None):
    """This function trains a separate EVM model for each class in the data.

    This function computes the EVM models from the given training data.

    Parameters
    ----------

    data: [2D :py:class:`numpy.ndarray`] or [[1D :py:class:`numpy.ndarray`]]
      The list of training data.
      Each element in the list contains the training data of one class, in the order `[number_of_features, feature_dimension]`.
      The `feature_dimension` across all features must be identical.

    parallel: int or ``None``
      If specified, use the given number of parallel processes to train EVM models.
      Otherwise, each EVM is computed in sequence.
    """
    self._evms = None
    # first, perform clustering
    if self.cluster_centers is not None:
      centers = self._cluster_data(data, parallel)
    else:
      centers = None
    # now, compute EVMs
    self._evms = self._train_evms(data, centers, parallel)


  def probabilities(self, points, parallel=None):
    """Computes the probabilities for all EVMs for the given data points.

    Parameters
    ----------

    points: [:py:class:`numpy.ndarray`]
      The points, for which to compute the probability.
      Each point needs to have the ``shape`` according to :py:attr:`shape`, which is identical to shape of the the training set features.

    parallel : int ``None``
      If given, use this number of parallel processes for processing.
      The number of parallel processes is limited by the number of extreme vectors :py:attr:`size`.

    Returns
    -------

    [[[float]]] : a three-dimensional list of probabilities for each point, each evm and each extreme vector inside the evm.
      Indices for the probabilities are ``(point, evm, extreme_vector)``.
    """
    if parallel is None:
      probs = [_probabilities(evm, points) for evm in self._evms]
    else:
      global _evms, _points
      _evms = self._evms
      _points = points
      pool = multiprocessing.Pool(parallel)
      probs = pool.map(_probabilities_parallel, range(self.size))
      pool.close()
      pool.join()
    # re-arange such that the first index is the point, the second the evm and the third the extreme vector
    return [[probs[e][p] for e in range(self.size)] for p in range(len(points))]


  def max_probabilities(self, points = None, probabilities = None, parallel = None):
    """Computes the maximum probabilities and their accoring exteme vector for the given data points.

    Parameters
    ----------

    points: [:py:class:`numpy.ndarray`]
      The points, for which to compute the probability.
      Each point needs to have the ``shape`` according to :py:attr:`shape`, which is identical to shape of the the training set features.
      Can be omitted when the ``probabilities`` parameter is given.

    probabilities : [[[float]]] or ``None``
      The probabilities that were returned by the :py:meth:`probabilities` function.
      If not given, they are first computed from the given ``points``.

    parallel : int or ``None``
      If given, use this number of parallel processes to compute probabilities.
      The number of parallel processes is limited by the number of extreme vectors :py:attr:`size`.
      Ignored, when ``probabilities`` are given on command line.

    Returns
    -------

    [float]
      The maximum probability of inclusion for each of the points.

    [(int,int)]
      The list of tuples of indices into :py:attr:`evms` and their according `:py:attr:`EVM.extreme_vectors` for the given points.
    """
    # compute probabilities
    if probabilities is None: probabilities = self.probabilities(points, parallel)
    # get maximum probability per EVM
    indices = []
    for p in range(len(probabilities)):
      # compute maximum indices for all evs per evm
      max_per_ev = [numpy.argmax(probabilities[p][e]) for e in range(self.size)]
      max_per_evm = numpy.argmax([probabilities[p][e][max_per_ev[e]] for e in range(self.size)])
      indices.append((max_per_evm, max_per_ev[max_per_evm]))

    # return maximum probabilities and
    return [probabilities[i][e][m] for i,(e,m) in enumerate(indices)], indices


  def save(self, h5):
    """Saves this object to HDF5

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to save, or the (subgroup of the) HDF5 file opened for writing.
    """
    if self._evms is None:
      raise RuntimeError("The model has not been trained yet")
    # open file for writing; create if not existent
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'w')

    # write EVMs
    for i, evm in enumerate(self._evms):
      evm.save(h5.create_group("EVM-%d" % (i+1)))


  def load(self, h5):
    """Loads this object from HDF5.

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to load, or the (subgroup of the) HDF5 file opened for reading.
    """
    # open file for reading
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'r')

    # load evms
    self._evms = []
    i = 1
    while "EVM-%d" % i in h5:
      self._evms.append(EVM(h5["EVM-%d" % (i)], log_level='debug'))
      i += 1


  @property
  def size(self):
    """The number of EVM's."""
    if self._evms is not None:
      return len(self._evms)
    else:
      return 0

  @property
  def evms(self):
    """The EVM's, in the same order as the training classes."""
    return self._evms
