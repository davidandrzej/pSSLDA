# PARALLEL SEMI-SUPERVISED LDA (pSSLDA)

David Andrzejewski (andrzeje@cs.wisc.edu)
Department of Computer Sciences
University of Wisconsin-Madison, USA



This software implements an extension of Latent Dirichlet Allocation
(LDA) [2] which includes "topic-in-set knowledge", or z-labels [1],
allowing the user to supply (possibly noisy) labels for specific
latent topic assignments.  Parallelized inference is done by the
Approximate Distributed (AD) [3] collapsed Gibbs sampling algorithm.

This code can also be used to do parallel inference for "standard"
LDA.

The implementation consists of Python extension modules written in C
and Cython. Hannah Devinney ([hdevinney](https://github.com/hdevinney)) migrated the code to be Python3-compatible.


## BUILD/INSTALL

Building this module requires Python, NumPy, Cython, and a C compiler.
From the command-line, do:

`% python setup.py install`

(Note that if things are installed to non-standard locations, you may
need to make the appropriate changes in setup.py)

There is a simple example scipt showing how to use pSSLDA:

`% python example/example_py3.py`



## LOCAL INSTALL

If you do not have write access to your Python installation directories,
you will need to tell setup.py to install this module somewhere else.
For example:

`% python setup.py install --prefix=~/local`

will install the module under a subdirectory of your home directory called 
"local".

It may then be necessary to let Python know where that is by setting
the `PYTHONPATH` environment variable (e.g., in `.bashrc` or `.cshrc`).  For
our example this might involve adding something like the line:

`setenv PYTHONPATH ~/local/lib/python2.5/site-packages`



## HOW TO USE

The commenting in the example.py script explains the meanings and
types of all input and return arguments.  The P parameter determines
how many parallel sampling processes to run - using a value larger
than the number of available cores is probably inadvisable.



## LICENSE

This software is open-source, released under the terms of the GNU
General Public License version 3, or any later version of the GPL (see
LICENSE).



## REFERENCES

[1] Andrzejewski, D. and Zhu, X. (2009).  Latent Dirichlet Allocation
with Topic-in-Set Knowledge. NAACL 2009 Workshop on Semi-supervised
Learning for NLP (NAACL-SSLNLP 2009)

[2] Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet
Allocation.  Journal of Machine Learning Research (JMLR) 3
(Mar. 2003), 993-1022.

[3] Newman, D., Asuncion, A., Smyth, P., and Welling, M.  Distributed
Algorithms for Topic Models. Journal of Machine Learning Research
(JMLR) 10 (Aug. 2009), 1801-1828.



### VERSION HISTORY
0.1     Initial release
