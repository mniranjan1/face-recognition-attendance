#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist

# Define package version
version = open("version.txt").read().rstrip()

setup(

    name='EVM',
    version=version,
    description='The Extreme Value Machine',
    url='http://bitbucket.org/vastlab/evm',
    license='BSD',
    author='Manuel Gunther',
    author_email='siebenkopf@googlemail.com',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires = open("requirements.txt").read().split(),

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

)
