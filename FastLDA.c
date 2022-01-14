/**
   Copyright (C) 2010 David Andrzejewski (andrzeje@cs.wisc.edu)
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "FastLDA.h"


/**
 * Do IN-PLACE z-label LDA Gibbs sample(s)
 * (z, localnw, and localnd will all be modified in-place)
 */
static PyObject* zLabelGibbs(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"zlabels","w","d","z","alpha","beta",
                           "localnw","localnd","globalnw",
                           "randseed","numsamp","verbose",NULL};
                           
  // Required args
  //
  PyObject* zlabel; // List of either: None, or dim-T NumPy weight array
  PyArrayObject* w; // NumPyArray of words
  PyArrayObject* d; // NumPyArray of doc indices
  PyArrayObject* z; // NumPyArray of topic assignments
  PyArrayObject* alpha; // NumPy Array
  PyArrayObject* beta; // NumPy Array
  // Counts
  PyArrayObject* localnw; // NumPy Array
  PyArrayObject* localnd; // NumPy Array
  PyArrayObject* globalnw; // NumPy Array
  // Parameters
  int randseed;
  // Optional
  int numsamp = 1;
  int verbose = 0;

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!O!O!O!O!O!i|ii",kwlist,
                                  &PyList_Type,&zlabel,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&z,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &PyArray_Type,&localnw,
                                  &PyArray_Type,&localnd,
                                  &PyArray_Type,&globalnw,
                                  &randseed,&numsamp,&verbose))
    // ERROR - bad args
    return NULL;

  // Init random number generator
  //
  srand((unsigned int) randseed);

  // More init...
  int T = PyArray_DIM(beta,0);
  int N = PyArray_DIM(w,0);
  // Precompute some sums
  PyArrayObject* localnw_colsum = (PyArrayObject*) PyArray_Sum(localnw,
                                                0,PyArray_INT,NULL);
  PyArrayObject* globalnw_colsum = (PyArrayObject*) PyArray_Sum(globalnw,
                                                  0,PyArray_INT,NULL);
  PyArrayObject* betasum = (PyArrayObject*) PyArray_Sum(beta,1,
                                                        PyArray_DOUBLE,NULL);

  // Temporary arrays used for sampling
  double* num = malloc(sizeof(double)*T);
  PyArrayObject* zlweights = NULL;

  // Use Gibbs sampling to get a new z
  // sample, one position at a time
  //
  int i,j,di,wi,zi,si;
  for (si = 0; si < numsamp; si++)
    {
      if(verbose != 0)
        printf("Sample %d of %d\n", si, numsamp);
      for(i = 0; i < N; i++) 
        {
          di = *((int*)PyArray_GETPTR1(d,i));
          wi = *((int*)PyArray_GETPTR1(w,i));
          zi = *((int*)PyArray_GETPTR1(z,i));

          // remove from count matrices
          (*((int*)PyArray_GETPTR2(localnw,wi,zi)))--;
          (*((int*)PyArray_GETPTR2(localnd,di,zi)))--;
          (*((int*)PyArray_GETPTR1(localnw_colsum,zi)))--;
      	
          // For each topic, calculate numerators
          double norm_sum = 0;

          // Is there a z-label for this index?
          int hasZL = 0;
          if(PyList_GetItem(zlabel,i) != Py_None)
            {
              // Record that we have z-label(s) for this index 
              hasZL = 1;
              // Entry should be a dim-T NumPy array of weights
              zlweights = (PyArrayObject*) PyList_GetItem(zlabel,i);
            }

          // Get un-normalized sampling probabilities for each topic 
          for(j = 0; j < T; j++) 
            { 
              double alpha_j = *((double*)PyArray_GETPTR2(alpha,0,j));
              double beta_i = *((double*)PyArray_GETPTR2(beta,j,wi));
              double bsum = *((double*)PyArray_GETPTR1(betasum,j));
              double nwdenom = *((int*)PyArray_GETPTR1(localnw_colsum,j));
              nwdenom += *((int*)PyArray_GETPTR1(globalnw_colsum,j)) + bsum;
              // Calculate numerator for each topicp
              // (NOTE: alpha denom omitted, since same for all topics)
              double nwnum = *((int*)PyArray_GETPTR2(localnw,wi,j));
              nwnum += *((int*)PyArray_GETPTR2(globalnw,wi,j)) + beta_i;
              double ndnum = *((int*)PyArray_GETPTR2(localnd,di,j)) + alpha_j;
              num[j] = ndnum * (nwnum / nwdenom);

              // Consider z-label, if applicable
              if(hasZL != 0)
                num[j] *= exp(*((double*)PyArray_GETPTR1(zlweights,j)));

              // Keep running sum
              norm_sum += num[j];
            }
	
          // Draw a sample
          //      
          j = mult_sample(num,norm_sum);
	
          // Update *local* count/cache matrices and sample vec
          //
          *((int*)PyArray_GETPTR1(z,i)) = j;
          (*((int*)PyArray_GETPTR2(localnw,wi,j)))++;
          (*((int*)PyArray_GETPTR2(localnd,di,j)))++;
          (*((int*)PyArray_GETPTR1(localnw_colsum,j)))++;
        }
    }

  // Memory cleanup
  //
  free(num);
  Py_DECREF(localnw_colsum);
  Py_DECREF(globalnw_colsum);

  // All changes done in-place on z and count matrices
  // (but need to INCREF Py_None, since caller gets a Py_None
  // which will be DECREF'ed when they're done with it...)
  Py_INCREF(Py_None); 
  return Py_None;  
}

/**
 * Do IN-PLACE Gibbs sample(s)
 * (z, localnw, and localnd will all be modified in-place)
 */
static PyObject* standardGibbs(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","d","z","alpha","beta",
                           "localnw","localnd","globalnw",
                           "randseed","numsamp","verbose",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPyArray of words
  PyArrayObject* d; // NumPyArray of doc indices
  PyArrayObject* z; // NumPyArray of topic assignments
  PyArrayObject* alpha; // NumPy Array
  PyArrayObject* beta; // NumPy Array
  // Counts
  PyArrayObject* localnw; // NumPy Array
  PyArrayObject* localnd; // NumPy Array
  PyArrayObject* globalnw; // NumPy Array
  // Parameters
  int randseed;
  // Optional
  int numsamp = 1;
  int verbose = 0;

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!O!O!O!O!i|ii",kwlist,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&z,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &PyArray_Type,&localnw,
                                  &PyArray_Type,&localnd,
                                  &PyArray_Type,&globalnw,
                                  &randseed,             
                                  &numsamp, &verbose))
    // ERROR - bad args
    return NULL;
 
  // Init random number generator
  //
  srand((unsigned int) randseed);

  // More init...
  int T = PyArray_DIM(beta,0);
  int N = PyArray_DIM(w,0);
  // Precompute some sums
  PyArrayObject* localnw_colsum = (PyArrayObject*) PyArray_Sum(localnw,
                                                0,PyArray_INT,NULL);
  PyArrayObject* globalnw_colsum = (PyArrayObject*) PyArray_Sum(globalnw,
                                                  0,PyArray_INT,NULL);
  PyArrayObject* betasum = (PyArrayObject*) PyArray_Sum(beta,1,
                                                        PyArray_DOUBLE,NULL);

  // Temporary array used for sampling
  double* num = malloc(sizeof(double)*T);

  // Use Gibbs sampling to get a new z
  // sample, one position at a time
  //
  int i,j,di,wi,zi,si;
  for (si = 0; si < numsamp; si++)
    {
      if(verbose != 0)
        printf("Sample %d of %d\n", si, numsamp);
      for(i = 0; i < N; i++) 
        {
          di = *((int*)PyArray_GETPTR1(d,i));
          wi = *((int*)PyArray_GETPTR1(w,i));
          zi = *((int*)PyArray_GETPTR1(z,i));

          // remove from count matrices
          (*((int*)PyArray_GETPTR2(localnw,wi,zi)))--;
          (*((int*)PyArray_GETPTR2(localnd,di,zi)))--;
          (*((int*)PyArray_GETPTR1(localnw_colsum,zi)))--;
      	
          // For each topic, calculate numerators
          double norm_sum = 0;
          for(j = 0; j < T; j++) 
            { 
              double alpha_j = *((double*)PyArray_GETPTR2(alpha,0,j));
              double beta_i = *((double*)PyArray_GETPTR2(beta,j,wi));
              double bsum = *((double*)PyArray_GETPTR1(betasum,j));
              double nwdenom = *((int*)PyArray_GETPTR1(localnw_colsum,j));
              nwdenom += *((int*)PyArray_GETPTR1(globalnw_colsum,j)) + bsum;
              // Calculate numerator for each topicp
              // (NOTE: alpha denom omitted, since same for all topics)
              double nwnum = *((int*)PyArray_GETPTR2(localnw,wi,j));
              nwnum += *((int*)PyArray_GETPTR2(globalnw,wi,j)) + beta_i;
              double ndnum = *((int*)PyArray_GETPTR2(localnd,di,j)) + alpha_j;
              num[j] = ndnum * (nwnum / nwdenom);
              // Keep running sum
              norm_sum += num[j];
            }
	
          // Draw a sample
          //      
          j = mult_sample(num,norm_sum);
	
          // Update *local* count/cache matrices and sample vec
          //
          *((int*)PyArray_GETPTR1(z,i)) = j;
          (*((int*)PyArray_GETPTR2(localnw,wi,j)))++;
          (*((int*)PyArray_GETPTR2(localnd,di,j)))++;
          (*((int*)PyArray_GETPTR1(localnw_colsum,j)))++;
        }
    }

  // Memory cleanup
  //
  free(num);
  Py_DECREF(localnw_colsum);
  Py_DECREF(globalnw_colsum);

  // All changes done in-place on z and count matrices
  // (but need to INCREF Py_None, since caller gets a Py_None
  // which will be DECREF'ed when they're done with it...)
  Py_INCREF(Py_None); 
  return Py_None;
}


/**
 * Online construction of an initial z-sample
 */
static PyObject* onlineInit(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","d","alpha","beta","randseed",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPy Array of words
  PyArrayObject* d; // NumPy Array of doc indices
  PyArrayObject* alpha; // NumPy Array
  PyArrayObject* beta; // NumPy Array
  int randseed;

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!i",kwlist,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &randseed))
    // ERROR - bad args
    return NULL;
 
  // Init random number generator
  srand((unsigned int) randseed);
  
  // Get dimensionality info
  int T = PyArray_DIM(beta,0);
  int W = PyArray_DIM(beta,1);
  int N = PyArray_DIM(w,0);  
  //update for py3:
  int D = PyLong_AsLong(PyArray_Max(d,NPY_MAXDIMS,NULL)) + 1;

  // Pre-calculate beta sums
  PyArrayObject* betasum = (PyArrayObject*) PyArray_Sum(beta,1,
                                                        PyArray_DOUBLE,NULL);

  // Construct count matrices
  npy_intp* nwdims = malloc(sizeof(npy_intp)*2);
  nwdims[0] = W;
  nwdims[1] = T;
  PyArrayObject* nw = (PyArrayObject*) PyArray_ZEROS(2,nwdims,
                                                     PyArray_INT,0);
  PyArrayObject* nw_colsum = (PyArrayObject*) PyArray_Sum(nw,0,
                                                          PyArray_INT,NULL);
  free(nwdims);

  npy_intp* nddims = malloc(sizeof(npy_intp)*2);
  nddims[0] = D;
  nddims[1] = T;
  PyArrayObject* nd =  (PyArrayObject*) PyArray_ZEROS(2,nddims,
                                                      PyArray_INT,0);
  free(nddims);

  // Build init z sample, one word at a time
  //
  npy_intp* zdims = malloc(sizeof(npy_intp));
  zdims[0] = N;
  PyArrayObject* z = (PyArrayObject*) PyArray_ZEROS(1,zdims,PyArray_INT,0);
  free(zdims);

  // Temporary array used for sampling
  double* num = malloc(sizeof(double)*T);

  // For each word in the corpus
  int i,j,di,wi;
  for(i = 0; i < N; i++) 
    {
      di = *((int*)PyArray_GETPTR1(d,i));
      wi = *((int*)PyArray_GETPTR1(w,i));
	      
      // For each topic, calculate numerators
      double norm_sum = 0;
      for(j = 0; j < T; j++) 
        { 
          double alpha_j = *((double*)PyArray_GETPTR2(alpha,0,j));
          double beta_i = *((double*)PyArray_GETPTR2(beta,j,wi));
          double bsum = *((double*)PyArray_GETPTR1(betasum,j));	
          double denom_1 = *((int*)PyArray_GETPTR1(nw_colsum,j)) + bsum;

          // Calculate numerator for this topic
          // (NOTE: alpha denom omitted, since same for all topics)
          num[j] = ((*((int*)PyArray_GETPTR2(nw,wi,j)))+beta_i) / denom_1;
          num[j] = num[j] * (*((int*)PyArray_GETPTR2(nd,di,j))+alpha_j);

          norm_sum += num[j];
        }
	
      // Draw a sample
      //   
      j = mult_sample(num,norm_sum);
	
      // Update count/cache matrices and initial sample vec
      //
      *((int*)PyArray_GETPTR1(z,i)) = j;

      (*((int*)PyArray_GETPTR2(nw,wi,j)))++;
      (*((int*)PyArray_GETPTR2(nd,di,j)))++;
      (*((int*)PyArray_GETPTR1(nw_colsum,j)))++;
    }

  // Cleanup and return z *without* doing INCREF
  //
  Py_DECREF(betasum);
  Py_DECREF(nw_colsum);
  Py_DECREF(nw);
  Py_DECREF(nd);
  free(num);
  return Py_BuildValue("N",z);  
}

/**
 * Construct count matrices nw (W x T) and nd (D x T)
 */
static PyObject* countMatrices(PyObject* self, PyObject* args, 
                               PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","W","d","D","z","T",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPyArray of words
  int W; // vocab size
  PyArrayObject* d; // NumPyArray of doc indices
  int D; // number of docs
  PyArrayObject* z; // NumPyArray of topic assignments
  int T; // number of topics

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!iO!iO!i",kwlist,
                                  &PyArray_Type,&w,&W,
                                  &PyArray_Type,&d,&D,
                                  &PyArray_Type,&z,&T))
    // ERROR - bad args
    return NULL;
  
  // counts will hold result (pointer to size-2 array of PyObject pointers)
  PyArrayObject** counts = malloc(sizeof(PyArrayObject*) * 2);
  _countMatrices(w,W,d,D,z,T,&counts);
  // Construct return value *without* INCREFing (caller now holds references)
  PyArrayObject* nw = counts[0];
  PyArrayObject* nd = counts[1];
  PyObject* retval = Py_BuildValue("NN",nw,nd);
  // Cleanup and return 
  free(counts);
  return retval;
}

/**
 * INTERNAL C VERSION
 */
static int _countMatrices(PyArrayObject* w, int W, PyArrayObject* d, int D,
                          PyArrayObject* z, int T, PyArrayObject*** counts)
{
  // Construct count matrices
  npy_intp* nwdims = malloc(sizeof(npy_intp)*2);
  nwdims[0] = W;
  nwdims[1] = T;
  PyArrayObject* nw = (PyArrayObject*) PyArray_ZEROS(2,nwdims,
                                                     PyArray_INT,0);
  free(nwdims);

  npy_intp* nddims = malloc(sizeof(npy_intp)*2);
  nddims[0] = D;
  nddims[1] = T;
  PyArrayObject* nd =  (PyArrayObject*) PyArray_ZEROS(2,nddims,
                                                      PyArray_INT,0);
  // Count for each word in the corpus
  int N = PyArray_DIM(w,0);
  int i,di,wi,zi;
  for(i = 0; i < N; i++) 
    {
      di = *((int*)PyArray_GETPTR1(d,i));
      wi = *((int*)PyArray_GETPTR1(w,i));
      zi = *((int*)PyArray_GETPTR1(z,i));

      (*((int*)PyArray_GETPTR2(nw,wi,zi)))++;
      (*((int*)PyArray_GETPTR2(nd,di,zi)))++;
    }

  // Put values in array pointed to by counts
  (*counts)[0] = nw;
  (*counts)[1] = nd;  
  return OK;
}

/**
 * Construct 'expected' count matrices nw (W x T) and nd (D x T)
 * for the case where we have relaxed/probabilistic/soft z-assign
 */
static PyObject* expectedCountMatrices(PyObject* self, PyObject* args, 
                                       PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","W","d","D","z","T",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPyArray of words
  int W; // vocab size
  PyArrayObject* d; // NumPyArray of doc indices
  int D; // number of docs
  PyArrayObject* z; // N x T NumPyArray of *soft* topic assignments
  int T; // number of topics

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!iO!iO!i",kwlist,
                                  &PyArray_Type,&w,&W,
                                  &PyArray_Type,&d,&D,
                                  &PyArray_Type,&z,&T))
    // ERROR - bad args
    return NULL;

  // Construct count matrices
  npy_intp* nwdims = malloc(sizeof(npy_intp)*2);
  nwdims[0] = W;
  nwdims[1] = T;
  PyArrayObject* nw = (PyArrayObject*) PyArray_ZEROS(2,nwdims,
                                                     PyArray_DOUBLE,0);
  free(nwdims);

  npy_intp* nddims = malloc(sizeof(npy_intp)*2);
  nddims[0] = D;
  nddims[1] = T;
  PyArrayObject* nd =  (PyArrayObject*) PyArray_ZEROS(2,nddims,
                                                      PyArray_DOUBLE,0);
  // Count for each word in the corpus
  int N = PyArray_DIM(w,0);
  int i,di,wi,zi;
  double pzi;
  for(i = 0; i < N; i++) 
    {
      // doc/word for this index
      di = *((int*)PyArray_GETPTR1(d,i));
      wi = *((int*)PyArray_GETPTR1(w,i));
      // count weighted/relaxed values for each topic
      for(zi = 0; zi < T; zi++)
        {          
          // soft assignment value for this index-topic
          pzi = *((double*)PyArray_GETPTR2(z,i,zi));
          // add to expected counts
          (*((double*)PyArray_GETPTR2(nw,wi,zi))) += pzi;
          (*((double*)PyArray_GETPTR2(nd,di,zi))) += pzi;
        }
    }

  // Return *without* INCREFing (caller now holds reference)
  return Py_BuildValue("NN",nw,nd);
}

/**
 * Estimate phi/theta as the mean of the posterior
 */
static PyObject* estPhiTheta(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"nw","nd","alpha","beta",NULL};
                           
  // Required args
  //
  PyArrayObject* nw; // NumPyArray of words-topic counts
  PyArrayObject* nd; // NumPyArray of doc-topic counts
  PyArrayObject* alpha; // NumPyArray 
  PyArrayObject* beta; // NumPyArray 

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!",kwlist,
                                  &PyArray_Type,&nw,
                                  &PyArray_Type,&nd,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta))

    // ERROR - bad args
    return NULL;

  // Get dimensionality info
  int T = PyArray_DIM(nw,1);
  int W = PyArray_DIM(nw,0);
  int D = PyArray_DIM(nd,0);
  int d,t,w;    

  //  
  // theta
  //

  // Pre-calculate some useful sums
  PyArrayObject* nd_rowsum = (PyArrayObject*) PyArray_Sum(nd,1,
                                                          PyArray_DOUBLE,NULL);
  double alphasum = 0;
  for(t = 0; t < T; t++)
    alphasum += *((double*)PyArray_GETPTR2(alpha,0,t));
  
  // Construct theta
  npy_intp* tdims = malloc(sizeof(npy_intp)*2);
  tdims[0] = D;
  tdims[1] = T;
  PyArrayObject* theta = (PyArrayObject*) 
    PyArray_ZEROS(2,tdims,PyArray_DOUBLE,0);
  free(tdims);

  // Calculate theta
  for(d = 0; d < D; d++) 
    {
      double rowsum = *((double*)PyArray_GETPTR1(nd_rowsum,d));
      for(t = 0; t < T; t++)
        {
          double alpha_t = *((double*)PyArray_GETPTR2(alpha,0,t));
          int ndct = *((int*)PyArray_GETPTR2(nd,d,t));
          // Calc and assign theta entry
          double newval = (ndct + alpha_t) / (rowsum + alphasum);
          *((double*)PyArray_GETPTR2(theta,d,t)) = newval;
        }
    }

  //
  // phi
  //

  // Pre-calculate some useful sums
  PyArrayObject* betasum = (PyArrayObject*) PyArray_Sum(beta,1,
                                                        PyArray_DOUBLE,NULL);
  PyArrayObject* nw_colsum = (PyArrayObject*) PyArray_Sum(nw,0,
                                                          PyArray_INT,NULL);

  // Construct phi
  npy_intp* pdims = malloc(sizeof(npy_intp)*2);
  pdims[0] = T;
  pdims[1] = W;
  PyArrayObject* phi = (PyArrayObject*) 
    PyArray_ZEROS(2,pdims,PyArray_DOUBLE,0);
  free(pdims);

  // Calculate phi
  for(t = 0; t < T; t++) 
    {
      int colsum = (*((int*)PyArray_GETPTR1(nw_colsum,t)));
      double bsum = *((double*)PyArray_GETPTR1(betasum,t));
      for(w = 0; w < W; w++) 
        {
          double beta_w = *((double*)PyArray_GETPTR2(beta,t,w));
          int nwct = *((int*)PyArray_GETPTR2(nw,w,t));
          double newval = (beta_w + nwct) / (bsum + colsum);
          *((double*)PyArray_GETPTR2(phi,t,w)) = newval;
        }
    }

  // Cleanup
  Py_DECREF(nw_colsum);
  Py_DECREF(nd_rowsum);
  Py_DECREF(betasum);

  // Return *without* INCREFing (caller now holds reference)
  return Py_BuildValue("NN",phi,theta);
}

/**
 * MAP estimate of phi/theta from expected count matrices
 * (note that expected count matrices must be double/float, not int)
 *
 * Also, true MAP estimate would be 0 for (count + hyperparam) < 1, but
 * for numerial sanity we bound away from 0 by MIN_PHI / MIN_THETA
 */
static PyObject* mapPhiTheta(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"nw","nd","alpha","beta",NULL};
                           
  // Required args
  //
  PyArrayObject* nw; // NumPyArray of *expected* words-topic counts
  PyArrayObject* nd; // NumPyArray of *expected* doc-topic counts
  PyArrayObject* alpha; // NumPyArray 
  PyArrayObject* beta; // NumPyArray 

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!",kwlist,
                                  &PyArray_Type,&nw,
                                  &PyArray_Type,&nd,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta))

    // ERROR - bad args
    return NULL;

  // phitheta will hold result (pointer to size-2 array of PyObject pointers)
  PyArrayObject** phitheta = malloc(sizeof(PyArrayObject*) * 2);
  _mapPhiTheta(nw,nd,alpha,beta,&phitheta);
  PyArrayObject* phi = phitheta[0];
  PyArrayObject* theta = phitheta[1];
  // Build result *without* INCREFing (caller now holds reference)
  PyObject* retval = Py_BuildValue("NN",phi,theta);
  // Cleanup and return
  free(phitheta);
  return retval;
}

/**
 * INTERNAL C VERSION
 */
static int _mapPhiTheta(PyArrayObject* nw,  PyArrayObject* nd,
                        PyArrayObject* alpha, PyArrayObject* beta,
                        PyArrayObject*** phitheta)
{
  // Get dimensionality info
  int T = PyArray_DIM(nw,1);
  int W = PyArray_DIM(nw,0);
  int D = PyArray_DIM(nd,0);
  int d,t,w;    

  //  
  // theta
  //

  // Construct theta
  npy_intp* tdims = malloc(sizeof(npy_intp)*2);
  tdims[0] = D;
  tdims[1] = T;
  PyArrayObject* theta = (PyArrayObject*) 
    PyArray_ZEROS(2,tdims,PyArray_DOUBLE,0);
  free(tdims);

  // Calculate theta
  for(d = 0; d < D; d++) 
    {
      double normsum = 0;
      for(t = 0; t < T; t++)
        {
          double alpha_t = *((double*)PyArray_GETPTR2(alpha,0,t));
          double nd_dt = *((double*)PyArray_GETPTR2(nd,d,t));
          // Calc and assign theta entry
          double val = max(MIN_THETA, alpha_t + nd_dt - 1);
          *((double*)PyArray_GETPTR2(theta,d,t)) = val;
          normsum += val;
        }
      // normalize
      for(t = 0; t < T; t++)
        {
          *((double*)PyArray_GETPTR2(theta,d,t)) /= normsum;
        }
    }

  //
  // phi
  //

  // Construct phi
  npy_intp* pdims = malloc(sizeof(npy_intp)*2);
  pdims[0] = T;
  pdims[1] = W;
  PyArrayObject* phi = (PyArrayObject*) 
    PyArray_ZEROS(2,pdims,PyArray_DOUBLE,0);
  free(pdims);

  // Calculate phi
  for(t = 0; t < T; t++) 
    {
      double normsum = 0;
      for(w = 0; w < W; w++) 
        {
          double beta_tw = *((double*)PyArray_GETPTR2(beta,t,w));
          double nw_wt = *((double*)PyArray_GETPTR2(nw,w,t));
          // Calc and assign phi entry
          double val = max(MIN_PHI, beta_tw + nw_wt - 1);
          *((double*)PyArray_GETPTR2(phi,t,w)) = val;
          normsum += val;
        }
      // normalize
      for(w = 0; w < W; w++) 
        {
          *((double*)PyArray_GETPTR2(phi,t,w)) /= normsum;
        }
    }

  // Return as C array of PyArrayObject pointers
  (*phitheta)[0] = phi;
  (*phitheta)[1] = theta;
  return OK;
}

/**
 * Calculate LDA logike of (z,phi,theta) given (w,alpha,beta)
 */
static PyObject* ldaLoglike(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","d","z","phi","theta","alpha","beta",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPyArray of words
  PyArrayObject* d; // NumPyArray of doc indices
  PyArrayObject* z; // NumPyArray of topic assignments
  PyArrayObject* phi; // NumPyArray of topic-word probs
  PyArrayObject* theta; // NumPyArray of doc-topic probs
  PyArrayObject* alpha; // NumPyArray of doc-topic probs
  PyArrayObject* beta; // NumPyArray of doc-topic probs

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!O!O!O!",kwlist,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&z,
                                  &PyArray_Type,&phi,
                                  &PyArray_Type,&theta,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta))
    // ERROR - bad args
    return NULL;
  
  // Call internal C method
  double ll = _ldaLoglike(w,d,z,phi,theta,alpha,beta);
  // Package return value
  return Py_BuildValue("d",ll);
}

static double _ldaLoglike(PyArrayObject* w, PyArrayObject* d,
                          PyArrayObject* z,
                          PyArrayObject* phi, PyArrayObject* theta,
                          PyArrayObject* alpha, PyArrayObject*beta)
{
  // Get some dimensionalities
  int T = PyArray_DIM(phi,0);
  int W = PyArray_DIM(phi,1);
  int D = PyArray_DIM(theta,0);

  // Get count matrices
  PyArrayObject** counts = malloc(sizeof(PyArrayObject*) * 2);
  _countMatrices(w,W,d,D,z,T,&counts);
  PyArrayObject* nw = counts[0];
  PyArrayObject* nd = counts[1];
  free(counts);

  double ll = 0; 
  
  
  // DOES NOT WORK YET - NO EASY WAY TO GET A 
  // NUMPY ARRAY 'SLICE' FROM C...?
  
  /* int ti,di; */
  /* // Topic-word Dirichlet */
  /* for(ti = 0; ti < T; ti++) */
  /*   { */
      
  /*   } */

  /* // Topic-word multinomial */
  /* for(ti = 0; ti < T; ti++) */
  /*   { */

  /*   } */

  /* // Doc-topic Dirichlet */
  /* for(di = 0; di < D; di++) */
  /*   { */

  /*   } */

  /* // Doc-topic multinomial */
  /* for(di = 0; di < D; di++) */
  /*   { */

  /*   } */
  

  // Cleanup and return
  Py_DECREF(nw);
  Py_DECREF(nd);
  return ll;
}


/**
 * Multinomial log-likelihood of counts given multinomial parameters theta
 */
static double _logMult(PyArrayObject* counts, PyArrayObject* theta)
{
  int i;
  double ll = 0;
  for(i = 0; i < PyArray_DIM(counts,0); i++)
    ll += *((int*)PyArray_GETPTR1(counts,i)) * 
      log(*((double*)PyArray_GETPTR1(theta,i)));
  return ll;
}

/**
 * Dirichlet log-likelihood of multinomial params 
 * theta given hyperparameters alpha
 */
static double _logDir(PyArrayObject* theta, PyArrayObject* alpha)
{
  double ll = 0; 
  int i;
  double asum = 0;

  // Normalization term denominator
  for(i = 0; i < PyArray_DIM(alpha,0); i++)
    {
      asum += *((double*)PyArray_GETPTR1(alpha,i));
      ll -= lgamma(*((double*)PyArray_GETPTR1(alpha,i)));
    }
  // Normalization term numerator
  ll += lgamma(asum);
  // theta^(alpha-1) terms
  for(i = 0; i < PyArray_DIM(alpha,0); i++)
    {
      ll += (*((double*)PyArray_GETPTR1(alpha,i)) - 1) * 
        log(*((double*)PyArray_GETPTR1(theta,i)));
    }
  return ll;
}

/**
 * Calculate perplexity of (w,d) given (phi,theta)
 */
static PyObject* perplexity(PyObject* self, PyObject* args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"w","d","phi","theta",NULL};
                           
  // Required args
  //
  PyArrayObject* w; // NumPyArray of words
  PyArrayObject* d; // NumPyArray of doc indices
  PyArrayObject* phi; // NumPyArray of topic-word probs
  PyArrayObject* theta; // NumPyArray of doc-topic probs

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!",kwlist,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&phi,
                                  &PyArray_Type,&theta))
    // ERROR - bad args
    return NULL;
  
  // Get dimensionalities
  int T = PyArray_DIM(phi,0);
  int N = PyArray_DIM(w,0);

  int i,t,wi,di;
  double perplex = 0;
  for(i = 0; i < N; i++)    
    {
      wi = *((int*)PyArray_GETPTR1(w,i));
      di = *((int*)PyArray_GETPTR1(d,i));
      double dot = 0;
      for(t = 0; t < T; t++)
        {
          dot += (*((double*)PyArray_GETPTR2(phi,t,wi))) *
            (*((double*)PyArray_GETPTR2(theta,di,t)));
        }
      perplex += log(dot);
    }
  perplex /= N;

  // Return avg perplexity
  return Py_BuildValue("d",perplex);
}

/**
 * Draw a multinomial sample propto vals
 * (!!! assumes sum is the correct sum for vals !!!)
 */
static int mult_sample(double* vals, double norm_sum)
{
  double rand_sample = unif() * norm_sum;
  double tmp_sum = 0;
  int j = 0;
  while(tmp_sum < rand_sample || j == 0) {
    tmp_sum += vals[j];
    j++;
  }
  return j - 1;
}

//
// PYTHON EXTENSION BOILERPLATE BELOW
//

// Module method table
PyMethodDef methods[] = 
  {
    {"zLabelGibbs", (PyCFunction) zLabelGibbs,
     METH_VARARGS | METH_KEYWORDS, 
     "Take a single in-place z-label LDA Gibbs sample"},
    {"standardGibbs", (PyCFunction) standardGibbs, 
     METH_VARARGS | METH_KEYWORDS, 
     "Take a single in-place standard LDA Gibbs sample"},
    {"onlineInit", (PyCFunctionWithKeywords) onlineInit, //PyCFunction ?
    // METH_KEYWORDS, "Do online LDA z-initialization"}, //Bad flag error
     METH_VARARGS | METH_KEYWORDS, //SegFault
    // METH_VARARGS, //SegFault
    // METH_FASTCALL,
     "Do online LDA z-initialization"},
    {"expectedCountMatrices", (PyCFunction) expectedCountMatrices, 
     //     METH_KEYWORDS, 
     METH_VARARGS | METH_KEYWORDS, 
     "Construct 'expected' nw/nd count matrices from relaxed z-assignments"},
    {"countMatrices", (PyCFunction) countMatrices, 
     //     METH_KEYWORDS, "Construct nw/nd count matrices"},
     METH_VARARGS | METH_KEYWORDS, "Construct nw/nd count matrices"},
    {"mapPhiTheta", (PyCFunction) mapPhiTheta,
	//     METH_KEYWORDS, "MAP estimate of phi/theta from expected count matrices"},
     METH_VARARGS |  METH_KEYWORDS, "MAP estimate of phi/theta from expected count matrices"},
    {"estPhiTheta", (PyCFunction) estPhiTheta,
	//     METH_KEYWORDS, "Estimate phi/theta from count matrices"},
     METH_VARARGS | METH_KEYWORDS, "Estimate phi/theta from count matrices"},
    {"perplexity", (PyCFunction) perplexity,
	//     METH_KEYWORDS, "Calc per-word perplexity given phi/theta"},
     METH_VARARGS | METH_KEYWORDS, "Calc per-word perplexity given phi/theta"},
    {"ldaLoglike", (PyCFunction) ldaLoglike,
	//     METH_KEYWORDS, "Calc LDA logike of (z,phi,theta) given (w,alpha,beta)"},
     METH_VARARGS | METH_KEYWORDS, "Calc LDA logike of (z,phi,theta) given (w,alpha,beta)"},
    {NULL, NULL, 0, NULL}  // Is a 'sentinel' (?)
  };

// This is a macro that does stuff for us (linkage, declaration, etc)
//python3 compatible: 2 parts.
static struct PyModuleDef FastLDA = {
  PyModuleDef_HEAD_INIT,
  "FastLDA", //module name
  "python3-compatible version", //documentation
  -1, //"-1 if the module keeps state in global variables"
  methods
};

PyMODINIT_FUNC PyInit_FastLDA(void)
{
  import_array(); //NumPy satisfaction DO NOT MOVE FROM FIRST LINE/HERE
  return PyModule_Create(&FastLDA);
}

