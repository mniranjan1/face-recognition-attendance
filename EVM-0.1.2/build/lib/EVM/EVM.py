import numpy
import scipy.spatial
import libmr
import h5py
import pickle

import logging
logger = logging.getLogger("evm")

import multiprocessing


def _distance(positive, negatives, distance_function):
  return [distance_function(positive, n) for n in negatives]

def _distance_parallel(args):
  p, distance_function = args
  global _positives, _negatives
  return _distance(_positives[p], _negatives, distance_function)

def _distance_cover(args):
  p, distance_function = args
  global _positives
  return _distance(_positives[p], _positives[p+1:], distance_function)


def _fit_weibull(distances, tailsize, distance_multiplier, low):
  mr = libmr.MR()
  if low:
    mr.fit_low([d*distance_multiplier for d in distances], min(tailsize, len(distances)))
  else:
    mr.fit_high(distances, min(tailsize, len(distances)-1)) # make sure to exclude the distance to itself
  return mr


def _fit_parallel(args):
  d, tailsize, distance_multiplier, low = args
  global _distances
  return _fit_weibull(_distances[d], tailsize, distance_multiplier, low)

def _probabilities(weibull, distances):
  return weibull.w_score_vector(numpy.array(distances))


def _compute_probabilities(p):
  global _margin_weibulls, _cover_weibulls, _distances
  # compute margin probabilities
  if _cover_weibulls is not None and _cover_weibulls[p].get_params()[0] > _margin_weibulls[p].get_params()[0]:
    # compute cover probabilities
    return 1. - _probabilities(_cover_weibulls[p], _distances[p])
  else:
    # compute margin probabilities
    return _probabilities(_margin_weibulls[p], _distances[p])



class EVM (object):

  """This class represents the Extreme Value Machine.

  The constructor can be called in two different ways:

  1. Creating an empty (untrained) machine by setting the ``tailsize`` to any positive integer.
     All other parameters can be set as well.
     Please :py:meth:`train` the machine before using it.

     Example::

       EVM(100, 0.5, distance_function = scipy.spatial.distance.euclidean)

  2. Loading a pre-trained machine by providing the filename of -- or the :py:class:`h5py.Group` inside -- the HDF5 file.
     All other parameters are ignored, as they are read from the file.

     Example::

       h5 = h5py.File("EVM.hdf5", 'r')
       EVM(h5["/some/group"])
  """

  def __init__(self,
    tailsize,
    cover_threshold = None,
    distance_multiplier = 0.5,
    distance_function = scipy.spatial.distance.cosine,
    include_cover_probability = False,
    log_level = 'info'
  ):
    self.log_level = log_level
    if isinstance(tailsize, (str, h5py.Group)):
      return self.load(tailsize)

    self.tailsize = tailsize
    self.cover_threshold = cover_threshold
    self.distance_function = distance_function
    self.distance_multiplier = distance_multiplier
    self.include_cover_probability = include_cover_probability

    self._positives = None
    self._margin_weibulls = None
    self._cover_weibulls = None
    self._extreme_vectors = None
    self._covered_vectors = None


  def _fit(self, distances, parallel, low = True):
    """Internal function to compute EVT fits from the given set of distances. Do not call directly."""
    if parallel is None:
      self.log("Performing Weibull fittings to %d positives", len(self._positives))
      weibulls = [_fit_weibull(d, self.tailsize, self.distance_multiplier, low) for d in distances]
    else:
      global _distances
      _distances = distances
      # create pool
      pool = multiprocessing.Pool(parallel)
      # perform weibull fitting
      self.log("Performing Weibull fittings to %d positives in %d parallel processes", len(self._positives), parallel)
      weibulls = pool.map(_fit_parallel, ((d, self.tailsize, self.distance_multiplier, low) for d in range(len(distances))))
      pool.close()
      pool.join()
    return weibulls


  def _distances(self, negatives, parallel):
    """Internal function to compute distances between positives and negatives. Do not call directly."""
    # compute distances between positives and negatives
    if parallel is None:
      self.log("Computing %d distances between %d positive and %d negative points", len(self._positives) * len(negatives), len(self._positives), len(negatives))
      distances = [_distance(p, negatives, self.distance_function) for p in self._positives]
    else:
      global _positives, _negatives
      # set data as global before creating the pool to avoid data copying;
      # this data is read-only in the parallel processes
      _positives = self._positives
      _negatives = negatives
      # create pool
      pool = multiprocessing.Pool(parallel)
      self.log("Computing %d distances between %d positive and %d negative points in %d parallel processes", len(self._positives) * len(negatives), len(self._positives), len(negatives), parallel)
      # compute distances
      distances = pool.map(_distance_parallel, ((p, self.distance_function) for p in range(len(self._positives))))
      pool.close()
      pool.join()

    return distances


  def _positive_distances(self, parallel):
    """Computes the distances between positives."""
    # first, compute only upper triangle
    N = len(self._positives)
    if parallel is None:
      upper_triangle = [_distance(self._positives[p], self._positives[p+1:], self.distance_function) for p in range(N-1)]
    else:
      global _positives
      _positives = self._positives
      pool = multiprocessing.Pool(parallel)
      upper_triangle = pool.map(_distance_cover, [(p, self.distance_function) for p in range(N-1)])
      pool.close()
      pool.join()

    # then, fill in matrix
    distances = numpy.zeros((N,N))
    for i in range(N-1):
      for j in range(N-1-i):
        distances[i, j+i+1] = distances[j+i+1, i] = upper_triangle[i][j]

    return distances


  def _set_cover(self, distances, parallel):
    """Internal function to deduce the model to keep only the most informant features (the so-called Extreme Vectors). Do not call directly."""
    N = len(self._positives)
    if self.cover_threshold is None or N <= 1:
      # no set cover to perform
      self._extreme_vectors = range(N)
      self._covered_vectors = [[i] for i in range(N)]
      return

    self.log("Computing set-cover for %d points", N)
    # compute distances between positives, if not given
    if distances is None:
      distances = self._positive_distances(parallel)

    # compute probabilities
    global _margin_weibulls, _cover_weibulls, _distances
    _margin_weibulls = self._margin_weibulls
    _cover_weibulls = self._cover_weibulls
    _distances = distances
    if parallel is None:
      probabilities = numpy.array([_compute_probabilities(i) for i in range(len(self._margin_weibulls))])
    else:
      pool = multiprocessing.Pool(parallel)
      probabilities = numpy.array(pool.map(_compute_probabilities, range(len(self._margin_weibulls))))
      pool.close()
      pool.join()

    # threshold by cover threshold
    thresholded = zip(*numpy.where(probabilities >= self.cover_threshold))

    # for each point, assign all other covered points
    # make sure that each point covers at least itself
    covering = [set((i,)) for i in range(N)]
    for x,y in thresholded:
      covering[x].add(y)

    # greedily add points that cover most of the others
    covered = set()
    universe = set(range(N))
    self._extreme_vectors = []
    self._covered_vectors = []
    while covered != universe:
      # get the point that covers most of the other points that are not yet covered
      ev = numpy.argmax([len(c - covered) for c in covering])
      # add it to the set of covered points
      self._extreme_vectors.append(ev)
      # add the covered points. note that a point might be covered by several EVs
      self._covered_vectors.append(sorted(covering[ev]))
      covered.update(covering[ev])
    self.log("Obtained %d extreme vectors", len(self._extreme_vectors))


  def train(self, positives, negatives=None, distances=None, parallel=None):
    """Trains the extreme value machine using the given positive samples of the class, and the negative samples that do not belong to the class.
    To speed up processing, an open :py:class:`multiprocessing.Pool` can be provided.

    .. warning::
       When providing the number of ``parallel`` processes, you can train only one EVM at the same time.
       Parallel processing is **not thread safe**.
       When training several EVM's in parallel, make sure that you leave ``parallel=None`` in this function.

    Parameters
    ----------

    positives : 2D :py:class:`numpy.ndarray` of float
      The points of the class to model.

    negatives : 2D :py:class:`numpy.ndarray` of float or ``None``
      Points of other classes, used to compute the distribution model.
      Ignored when ``distances`` are not ``None``.

    distances : 2D :py:class:`numpy.ndarray` of float or [[float]] or ``None``
      Distances between positives and negatives, used to compute the distribution model.
      If no distances are given, they are computed from the ``negatives``.
      A different number of distances can be provided for each of the ``positives``.

    parallel : int or ``None``
      If given, use this number of parallel processes.
      The number of parallel processes is limited by the number of ``positives``.
    """

    assert distances is not None or negatives is not None, "Either distances or negatives must not be `None`"

    # store all positives and their according Weibull distributions as the model
    self._positives = numpy.array(positives)

    # compute cover probability if wanted
    cover_distances = None
    if self.include_cover_probability:
      if len(positives) <= 3:
        logger.error("In order to compute cover probabilities, we need at least 3 positives, but we have only %d", len(positives))
        self.include_cover_probability = False
      else:
        # compute distances between all positives
        # these will include distances between same points, which will be removed from the weibull fit later
        # TODO: remove distances
        cover_distances = self._positive_distances(parallel)
        # and perform fit_high on these to get the cover weibulls
        self._cover_weibulls = self._fit(cover_distances, parallel, low=False)

    # now, train the margin probability
    if distances is not None:
      if negatives is not None:
        logger.warn("The negative points are ignored since distances are given directly")
      assert len(distances) == len(positives), "Distances need to be computed for all positive points"
    else:
      # first, train the weibull models for each positive point
      distances = self._distances(negatives, parallel)

    # compute weibulls from distances
    self._margin_weibulls = self._fit(distances, parallel)

    # then, perform model reduction using set-cover
    self._set_cover(cover_distances, parallel)


  def probabilities(self, points=None, distances=None, parallel=None):
    """Computes the probabilities for all extreme vectors for the given data points.

    Parameters
    ----------

    points: [:py:class:`numpy.ndarray`] or ``None``
      The points, for which to compute the probabilities.
      Each point needs to have the ``shape`` according to :py:attr:`shape`, which is identical to shape of the the training set features.
      Ignored if ``distances`` are given.

    distances: 2D :py:class:`numpy.ndarray` or ``None``
      The distances between points and the :py:attr:`extreme_vectors`, for which to compute the probabilities.
      Distances should be computed with the same distance function as used during training.
      If not given, distances are computed from the ``points``.

    parallel : int or ``None``
      If given, use this number of parallel processes.
      The number of parallel processes is limited by the number of extreme vectors :py:attr:`size`.

    Returns
    -------

    2D :py:class:`numpy.ndarray` of floats
      The probability of inclusion for each of the points to each of the extreme vectors.
      The array indices are in order `(point, extreme_vector)`.
    """

    assert distances is not None or points is not None, "Either distances or points must not be `None`"

    # check if distances are given
    if distances is not None:
      if points is not None:
        logger.warn("The negative points are ignored since distances are given directly")
      assert len(distances) == self.size, "Distances need to be computed for all extreme vectors"
    else:
      # compute distances
      if parallel is None:
        distances = [_distance(self._positives[e], points, self.distance_function) for e in self._extreme_vectors]
      else:
        # set global variables to avoid data copying before creating the pool
        global _positives, _negatives
        _positives = self.extreme_vectors
        _negatives = points
        pool = multiprocessing.Pool(parallel)
        distances = pool.map(_distance_parallel, ((p, self.distance_function) for p in range(self.size)))
        pool.close()
        pool.join()

    # set global variables to avoid data copying before creating a new pool
    global _margin_weibulls, _cover_weibulls, _distances
    _margin_weibulls = self.margin_weibulls
    _cover_weibulls = self.cover_weibulls
    _distances = distances
    # then, compute weibull scores
    if parallel is None:
      w_scores = [_compute_probabilities(i) for i in range(self.size)]
    else:
      pool = multiprocessing.Pool(parallel)
      w_scores = pool.map(_compute_probabilities, range(self.size))
      pool.close()
      pool.join()

    # finally, return the scores in corrected order
    return numpy.array(w_scores).transpose()


  def max_probabilities(self, points = None, distances = None, probabilities = None, parallel = None):
    """Computes the maximum probabilities and their accoring exteme vector for the given data points.

    Parameters
    ----------

    points: [:py:class:`numpy.ndarray`]
      The points, for which to compute the probability.
      Each point needs to have the ``shape`` according to :py:attr:`shape`, which is identical to shape of the the training set features.
      Ignored when ``distances`` or ``probabilities`` parameter is given.

    distances: 2D :py:class:`numpy.ndarray` or ``None``
      The distances between points and the :py:attr:`extreme_vectors`, for which to compute the probabilities.
      Distances should be computed with the same distance function as used during training.
      Ignored when ``probabilities`` are given.

    probabilities : 2D :py:class:`numpy.ndarray` of floats or ``None``
      The probabilities that were returned by the :py:meth:`probabilities` function.
      If not given, they are first computed from the given ``points`` or ``distances``.

    parallel : int or ``None``
      If given, use this number of parallel processes.
      The number of parallel processes is limited by the number of extreme vectors :py:attr:`size`.

    Returns
    -------

    [float]
      The maximum probability of inclusion for each of the points.

    [int]
      The list of indices into `:py:attr:`extreme_vectors` for the given points.
    """

    if probabilities is None: probabilities = self.probabilities(points, distances, parallel)
    indices = numpy.argmax(probabilities, axis=1)
    return [probabilities[i, j] for i,j in enumerate(indices)], indices


  def save(self, h5):
    """Saves this object to HDF5

    Parameters
    ----------

    h5 : ``str`` or :py:class:`h5py.File` or :py:class:`h5py.Group`
      The name of the file to save, or the (subgroup of the) HDF5 file opened for writing.
    """
    if self._positives is None:
      raise RuntimeError("The model has not been trained yet")
    # open file for writing; create if not existent
    if isinstance(h5, str):
      h5 = h5py.File(h5, 'w')
    # write features
    h5["Features"] = self._positives
    # write weibulls
    w = h5.create_dataset("MarginWeibulls", (len(self._margin_weibulls), len(self._margin_weibulls[0].as_binary())), numpy.uint8)
    for i in range(len(self._positives)):
      w[i] = self._margin_weibulls[i].as_binary()
    if self.include_cover_probability:
      c = h5.create_dataset("CoverWeibulls", (len(self._margin_weibulls), len(self._margin_weibulls[0].as_binary())), numpy.uint8)
      for i in range(len(self._positives)):
        c[i] = self._cover_weibulls[i].as_binary()
    else:
      self._cover_weibulls = None

    h5["ExtremeVectors"] = self._extreme_vectors
    # write covered indices
    for i in range(self.size):
      h5.create_dataset("Covered_%d" % (i+1), (len(self._covered_vectors[i]),), numpy.int32, self._covered_vectors[i])
    # write other parameteres (as attributes)
    w.attrs["Distance"] = pickle.dumps(self.distance_function)
    w.attrs["Tailsize"] = self.tailsize
    h5["ExtremeVectors"].attrs["CoverThreshold"] = self.cover_threshold if self.cover_threshold is not None else -1.


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
    # load features
    self._positives = h5["Features"][:]
    # load weibulls
    self._margin_weibulls = []
    w = h5["MarginWeibulls"]
    for i in range(w.shape[0]):
      self._margin_weibulls.append(libmr.load_from_binary(bytearray(w[i].tobytes())))
    self.include_cover_probability = "CoverWeibulls" in h5
    if self.include_cover_probability:
      self._cover_weibulls = []
      c = h5["CoverWeibulls"]
      for i in range(w.shape[0]):
        self._cover_weibulls.append(libmr.load_from_binary(bytearray(c[i].tobytes())))
      assert len(self._cover_weibulls) == len(self._margin_weibulls)

    # load extreme vectors
    e = h5["ExtremeVectors"]
    self._extreme_vectors = e[:]
    # load covered indices
    self._covered_vectors = [list(h5["Covered_%d" % (i+1)]) for i in range(self.size)]
    # load other parameteres
    self.distance_function = pickle.loads(w.attrs["Distance"])
    self.tailsize = w.attrs["Tailsize"]
    self.cover_threshold = e.attrs["CoverThreshold"]
    if self.cover_threshold == -1.: self.cover_threshold = None

  def log(self, *args, **kwargs):
    """Logs the given message using debug or info logging, see :py:attr:`log_level`"""
    {'info' : logger.info, 'debug' : logger.debug}[self.log_level](*args, **kwargs)


  @property
  def size(self):
    """The number of extreme vectors."""
    if self._extreme_vectors is not None:
      return len(self._extreme_vectors)
    else:
      return 0

  @property
  def shape(self):
    """The shape of the features that the :py:meth:`probability` function expects."""
    if self._positives is not None:
      return self._positives[0].shape

  @property
  def extreme_vectors(self):
    """The extreme vectors that this class stores."""
    if self._extreme_vectors is not None:
      return self._positives[self._extreme_vectors]

  @property
  def margin_weibulls(self):
    """The margin Weibull distributions for the extreme vectors."""
    if self._extreme_vectors is not None:
      return [self._margin_weibulls[e] for e in self._extreme_vectors]

  @property
  def cover_weibulls(self):
    """The cover Weibull distributions for the extreme vectors, if present."""
    if self._extreme_vectors is not None and self.include_cover_probability:
      return [self._cover_weibulls[e] for e in self._extreme_vectors]

  def covered(self, i):
    """Returns the vectors covered by the extreme vector with the given index"""
    if self._extreme_vectors is not None and i < self.size:
      return [self._positives[c] for c in self._covered_vectors[i]]
