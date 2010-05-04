/**
   Copyright (C) 2009 David Andrzejewski (andrzeje@cs.wisc.edu)
 
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
                           "randseed","numsamp",NULL};
                           
  // Required args
  //
  PyObject* zlabel; // List of (weight, ok Set) Tuples or None
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

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!O!O!O!O!O!i|i",kwlist,
                                  &PyList_Type,&zlabel,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&z,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &PyArray_Type,&localnw,
                                  &PyArray_Type,&localnd,
                                  &PyArray_Type,&globalnw,
                                  &randseed,&numsamp))
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
  double* zlweights = malloc(sizeof(double)*T);

  // Use Gibbs sampling to get a new z
  // sample, one position at a time
  //
  int i,j,di,wi,zi,si;
  for (si = 0; si < numsamp; si++)
    {
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
              hasZL = 1;
              PyObject* zltuple = PyList_GetItem(zlabel,i);
              double weight = PyFloat_AsDouble(PyTuple_GetItem(zltuple,0));
              PyObject* okset = PyTuple_GetItem(zltuple,1);
              for(j = 0; j < T; j++)
                {
                  if(PySet_Contains(okset,PyInt_FromLong(j)))
                    zlweights[j] = weight;                    
                  else
                    zlweights[j] = 0;
                }
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
                  num[j] *= exp(zlweights[j]);

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
  free(zlweights);
  free(num);
  Py_DECREF(localnw_colsum);
  Py_DECREF(globalnw_colsum);

  // All changes done in-place on z and count matrices
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
                           "randseed","numsamp",NULL};
                           
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

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!O!O!O!O!i|i",kwlist,
                                  &PyArray_Type,&w,
                                  &PyArray_Type,&d,
                                  &PyArray_Type,&z,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &PyArray_Type,&localnw,
                                  &PyArray_Type,&localnd,
                                  &PyArray_Type,&globalnw,
                                  &randseed,             
                                  &numsamp))
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
  int D = PyInt_AsLong(PyArray_Max(d,NPY_MAXDIMS,NULL)) + 1;

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
    {"onlineInit", (PyCFunction) onlineInit, 
     METH_KEYWORDS, "Do online LDA z-initialization"},
    {"countMatrices", (PyCFunction) countMatrices, 
     METH_KEYWORDS, "Construct nw/nd count matrices"},
    {"estPhiTheta", (PyCFunction) estPhiTheta,
     METH_KEYWORDS, "Estimate phi/theta from count matrices"},
    {"perplexity", (PyCFunction) perplexity,
     METH_KEYWORDS, "Calc per-word perplexity given phi/theta"},
    {NULL, NULL, 0, NULL}  // Is a 'sentinel' (?)
  };

// This is a macro that does stuff for us (linkage, declaration, etc)
PyMODINIT_FUNC 
initFastLDA() // Passes method table to init our module
{
  (void) Py_InitModule("FastLDA", methods); 
  import_array(); // Must do this to satisfy NumPy (!)
}
