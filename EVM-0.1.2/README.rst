The Extreme Value Machine
=========================

This package contains source code for the Extreme Value Machine (EVM).
The EVM is a model to compute probabilities for samples to belong to a certain class.
In opposition to other closed-set classification models, the EVM enables open-set classification, i.e., test samples might be of classes that are not contained in the training set.
Hence, the probability of this test sample should be low for any of the classes.

The EVM was introduced in the paper `The Extreme Value Machine <http://doi.org/10.1109/TPAMI.2017.2707495>`__.
If you are using the EVM in a scientific publication, please cite:

.. code-block:: latex

   @article{rudd2018evm,
     author={Rudd, Ethan M. and Jain, Lalit P. and Scheirer, Walter J. and Boult, Terrance E.},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
     title={The Extreme Value Machine},
     year={2018},
     volume={40},
     number={3},
     pages={762-768},
  }

Installation
------------

Pip
~~~

This package is on PyPI, so you can simply call:

.. code-block:: sh

   pip install evm

to install this package.
All required dependencies will be automatically installed, too.

Local build
~~~~~~~~~~~

The EVM package is using the setuptools installation package.
In order to compile the package locally, you can use:

.. code-block:: sh

   python setup.py build

which will install this package into the ``build/lib`` directory.
You can also call:

.. code-block:: sh

   python setup.py install

to install the package system-wide.


Usage
-----

This package provides mainly two classes, for different purposes.

Training a Single EVM
~~~~~~~~~~~~~~~~~~~~~

The first class is the ``EVM`` class, which holds all information that is required to model data from a specific class of your data.
This is done by computing distances from all instances of your class (``positives``) to instances of other classes (``negatives``).

First, you need to create an object of the ``EVM`` class.
Here, you need to specify several parameters:

* ``tailsize`` : An integral parameter that defines, how many negative samples are used to estimate the model parameters.
* ``cover_threshold`` : If not ``None`` (the default), specifies the probability threshold used to eliminate extreme vectors if they are covered by other extreme vectors with that probability. If ``None``, no model reduction will be performed.
* ``distance_multiplier`` : The multiplier to compute margin distances. ``0.5`` by default.
* ``distance_function`` : The distance function used to compute the distance between two samples; defaults to ``scipy.spatial.distance.cosine``
* ``include_cover_probabilities`` : (experimental) include cover probabilities that are calculated between samples of the same class. At least three samples per class are required to be able to compute cover probabilities.
* ``log_level`` : defines the verbosity of the EVM training. Possible values: ``'info'`` (the default), ``'debug'``.

.. code-block:: py

   import EVM, numpy, scipy
   evm = EVM.EVM(tailsize=10, cover_threshold = 0.7, distance_function=scipy.spatial.distance.euclidean)

After creating an ``EVM`` object, it needs to be trained.
You can train the model in two different ways.
In the most commonly used way, you pass both positive and negative data samples of your class and other classes, respectively:

.. code-block:: py

   class1 = numpy.random.normal((0,0),3,(50,2))
   class2 = numpy.random.normal((-10,10),3,(50,2))
   class3 = numpy.random.normal((10,-10),3,(50,2))

   evm.train(positives = class1, negatives = numpy.concatenate((class2, class3)))

Alternatively, you can also pre-compute the distances between all positive and all negative samples and pass the distance matrix in:

.. code-block:: py

   distances = scipy.spatial.distance.cdist(class1, numpy.concatenate((class2, class3)), 'euclidean')
   evm.train(positives = class1, distances = distances)

Now, you can compute the probability of any data point to belong to this class (actually, the function expects a list of points, so here we have to wrap it in square brackets):

.. code-block:: py

   probabilities = evm.probabilities([[0,0]])[0]

which will return the probability of inclusion for each of the extreme vectors.
Alternatively, you might be interested in the maximum probability, i.e., the probability that the test sample belongs to your class.
You can call:

.. code-block:: py

   probability, evm_index = evm.max_probabilities([[0,0]])

which will return the maximum probability over all extreme vectors, and the index of the extreme vectors that was the maximum.

Training Multi-Class EVMs
~~~~~~~~~~~~~~~~~~~~~~~~~

The second class that you can use to train EVMs is the ``MultipleEVM`` class.
For a given set of samples of several classes, it will compute an EVM model for each of the classes, taking all other classes as negatives.
The parameters are similar to the ``EVM`` class.

.. code-block:: py

   mevm = EVM.MultipleEVM(tailsize=10, cover_threshold = 0.7, distance_function=scipy.spatial.distance.euclidean)
   mevm.train([class1, class2, class3])

You can obtain the trained EVM models for each of the classes separately, if you want:

.. code-block:: py

   evm1 = mevm.evms[0]
   evm2 = mevm.evms[1]
   evm3 = mevm.evms[2]

For a given test sample, you can compute the probabilities for all extreme vectors in all EVMs by calling:

.. code-block:: py

   probabilities = mevm.probabilities([[0,0]])[0]

Similarly, you can compute the class with the maximum probability:

.. code-block:: py

   probabilities, indexes = mevm.max_probabilities([[0,0]])

where ``indexes`` contain both the index of the maximum class, and the index of the extreme vector inside that class.

Parallelism
~~~~~~~~~~~

Any of the public API functions to train or test ``EVM`` or ``MultipleEVM`` have a ``parallel`` parameter, where you can specify the maximum number of parallel threads to compute the function.

Example
~~~~~~~

In the root directory of this package, there is an ``example.py`` script that will show an exemplary usage of the ``EVM`` and ``MultipleEVM`` classes in order to display probabilities obtained on a sample 2D dataset.
You can run the example via

.. code-block:: sh

   python example.py

It might run for some time, and it will create a multi-page PDF file containing 4 plots:

1. Training a single EVM class from samples, and showing the extreme vectors.
2. Provide the probability of the class for inputs in a grid in color coding.
3. Training multiple EVM classes from the same samples, and showing the extreme vectors.
4. Provide the maximum probability of the classes for inputs in a grid in color coding.

In the color coding, black means that none of the classes had a high probability.
Since most of the regions are black, this indicates that the EVM has learned to classify unknown samples as unknown.
