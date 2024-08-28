from . import EVM, MultipleEVM
import numpy
import tempfile, os
import h5py
import scipy.spatial

numpy.random.seed(42)
positives = numpy.random.normal(-1,2,(100,50))
negatives = numpy.random.normal(1,2,(1000,50))
probes = numpy.random.normal(0,2,(10,50))


def test_no_cover():
  e = EVM(tailsize=30)
  assert e.size == 0
  assert e.shape is None
  assert e.extreme_vectors is None
  assert e.margin_weibulls is None
  assert e.cover_weibulls is None
  e.train(positives, negatives)
  assert e.size == 100
  assert e.shape == (50,)
  assert e.extreme_vectors.shape == (100,50)
  assert len(e.margin_weibulls) == 100
  assert e.cover_weibulls is None
  for i in range(e.size):
    assert numpy.allclose(e.covered(i), positives[i])

  probs = e.probabilities(probes)
  assert probs.shape == (10, 100)

  max_probs, max_index = e.max_probabilities(probabilities=probs)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, p in enumerate(max_index):
    assert max_probs[i] == probs[i,p]

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = EVM(fn)
    assert e2.size == 100, e2.size
    assert e2.shape == (50,), e2.shape

    probs2 = e2.probabilities(probes)
    assert numpy.allclose(probs, probs2)
    for i in range(e.size):
      assert numpy.allclose(e.covered(i), positives[i])
  finally:
    if os.path.exists(fn):
      os.remove(fn)

def test_cover_probability():
  e = EVM(tailsize=30, include_cover_probability=True)
  e.train(positives, negatives)
  assert e.size == 100, e.size
  assert e.shape == (50,), e.shape
  assert e.extreme_vectors.shape == (100,50)
  assert len(e.margin_weibulls) == 100
  assert len(e.cover_weibulls) == 100

  for i in range(e.size):
    assert any(numpy.allclose(e.extreme_vectors[i], covered) for covered in e.covered(i))

  probs = e.probabilities(probes)
  assert probs.shape == (10, 100)

  max_probs, max_index = e.max_probabilities(probabilities=probs)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, p in enumerate(max_index):
    assert max_probs[i] == probs[i,p]

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = EVM(tailsize=30)
    e2.load(fn)
    assert e2.size == 100, e2.size
    assert e2.shape == (50,), e2.shape

    probs2 = e2.probabilities(probes)
    assert numpy.allclose(probs, probs2)
    assert all(numpy.allclose(e.covered(i), e2.covered(i)) for i in range(e.size))
  finally:
    if os.path.exists(fn):
      os.remove(fn)


def test_set_cover():
  e = EVM(tailsize=30,cover_threshold=0.8)
  e.train(positives, negatives)
  assert e.size == 81, e.size
  assert e.shape == (50,), e.shape
  assert e.extreme_vectors.shape == (81,50)
  assert len(e.margin_weibulls) == 81

  for i in range(e.size):
    assert any(numpy.allclose(e.extreme_vectors[i], covered) for covered in e.covered(i))

  probs = e.probabilities(probes)
  assert probs.shape == (10, 81)

  max_probs, max_index = e.max_probabilities(probabilities=probs)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, p in enumerate(max_index):
    assert max_probs[i] == probs[i,p]

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = EVM(tailsize=30)
    e2.load(fn)
    assert e2.size == 81, e2.size
    assert e2.shape == (50,), e2.shape

    probs2 = e2.probabilities(probes)
    assert numpy.allclose(probs, probs2)
    assert all(numpy.allclose(e.covered(i), e2.covered(i)) for i in range(e.size))
  finally:
    if os.path.exists(fn):
      os.remove(fn)


def test_set_cover_parallel():
  e = EVM(tailsize=30,cover_threshold=0.8)
  e.train(positives, negatives, parallel=5)
  assert e.size == 81
  assert e.shape == (50,)
  assert e.extreme_vectors.shape == (81,50)
  assert len(e.margin_weibulls) == 81
  probs = e.probabilities(probes, parallel=5)
  assert probs.shape == (10, 81)

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  h5 = h5py.File(fn)
  try:
    grp = h5.create_group("Group")
    e.save(grp)
    h5.close()

    h5 = h5py.File(fn)
    e2 = EVM(h5["/Group"])
    assert e2.size == 81, e2.size
    assert e2.shape == (50,), e2.shape

    probs2 = e2.probabilities(probes)
    assert numpy.allclose(probs, probs2)
  finally:
    try:
      h5.close()
    except:
      pass
    os.remove(fn)


def test_set_cover_cover_parallel():
  e = EVM(tailsize=30,cover_threshold=0.8,include_cover_probability=True)
  e.train(positives, negatives,parallel=5)
  assert e.size == 3, e.size
  assert e.shape == (50,), e.shape
  assert e.extreme_vectors.shape == (3,50)
  assert len(e.margin_weibulls) == 3
  assert len(e.cover_weibulls) == 3

  for i in range(e.size):
    assert any(numpy.allclose(e.extreme_vectors[i], covered) for covered in e.covered(i))

  probs = e.probabilities(probes,parallel=5)
  assert probs.shape == (10, 3)

  max_probs, max_index = e.max_probabilities(probabilities=probs,parallel=5)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, p in enumerate(max_index):
    assert max_probs[i] == probs[i,p]

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = EVM(tailsize=30)
    e2.load(fn)
    assert e2.size == 3, e2.size
    assert e2.shape == (50,), e2.shape

    probs2 = e2.probabilities(probes,parallel=5)
    assert numpy.allclose(probs, probs2)
    assert all(numpy.allclose(e.covered(i), e2.covered(i)) for i in range(e.size))
  finally:
    if os.path.exists(fn):
      os.remove(fn)


def test_distances():
  e = EVM(tailsize=30,cover_threshold=0.8)
  distances = [[scipy.spatial.distance.cosine(p, n) for n in negatives] for p in positives]
  e.train(positives, distances=distances)
  assert e.size == 81
  assert e.shape == (50,)
  assert e.extreme_vectors.shape == (81,50)
  assert len(e.margin_weibulls) == 81
  distances = [[scipy.spatial.distance.cosine(p, n) for n in probes] for p in e.extreme_vectors]
  probs = e.probabilities(distances=distances)
  assert probs.shape == (10, 81)

  max_probs, max_index = e.max_probabilities(distances=distances)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, p in enumerate(max_index):
    assert max_probs[i] == probs[i,p]

def test_multiple():
  numpy.random.seed(42)
  data = [numpy.random.normal(i,2,(40,50)) for i in range(20)]
  probes = numpy.random.normal(8.,2,(10,50))

  e = MultipleEVM(tailsize=30,cover_threshold=0.8)
  e.train(data)
  assert e.size == 20

  probs = e.probabilities(probes)
  assert len(probs) == 10
  assert len(probs[0]) == 20

  max_probs, max_index = e.max_probabilities(probabilities=probs)
  assert len(max_probs) == 10
  assert len(max_index) == 10
  for i, (p,n) in enumerate(max_index):
    assert max_probs[i] == probs[i][p][n]


  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = MultipleEVM(tailsize=30)
    e2.load(fn)
    assert e2.size == 20

    probs2 = e2.probabilities(probes)
    assert all(numpy.allclose(p, p2) for pp,pp2 in zip(probs, probs2) for p,p2 in zip(pp,pp2))
  finally:
    if os.path.exists(fn):
      os.remove(fn)


def test_cluster():
  numpy.random.seed(42)
  data = [numpy.random.normal(i,2,(40,50)) for i in range(20)]
  probes = numpy.random.normal(8.,2,(10,50))

  e = MultipleEVM(tailsize=30,cover_threshold=0.8,cluster_centers=5)
  e.train(data, parallel=5)
  assert e.size == 20

  probs = e.probabilities(probes, parallel=5)
  assert len(probs) == 10
  assert len(probs[0]) == 20

  fh, fn = tempfile.mkstemp(prefix="evm", suffix=".hdf5")
  os.close(fh)
  try:
    e.save(fn)

    e2 = MultipleEVM(tailsize=30)
    e2.load(fn)
    assert e2.size == 20

    probs2 = e.probabilities(probes)
    assert all(numpy.allclose(p, p2) for pp,pp2 in zip(probs, probs2) for p,p2 in zip(pp,pp2))
  finally:
    if os.path.exists(fn):
      os.remove(fn)
