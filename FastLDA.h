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
#include <Python.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <numpy/arrayobject.h>

// Uniform rand between [0,1] (inclusive)
#define unif() ((double) rand()) / ((double) RAND_MAX)

#ifndef max
#define max(a,b) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#endif

// Function return codes
#define OK 0
#define FAIL 1

// Minimum values for phi/theta MAP estimates 
// (to avoid numerical issues with 0 values)
#define MIN_PHI 0.0001
#define MIN_THETA 0.0001

// Do a single IN-PLACE Gibbs sample for standard LDA
static PyObject* standardGibbs(PyObject* self, PyObject* args, PyObject* keywds);

// Do a single IN-PLACE Gibbs sample for z-label LDA
static PyObject* zLabelGibbs(PyObject* self, PyObject* args, PyObject* keywds);

// Online construction of an initial z-sample
static PyObject* onlineInit(PyObject* self, PyObject* args, PyObject* keywds);

// Build nw and nd count matrices
static PyObject* countMatrices(PyObject* self, PyObject* args, PyObject* keywds);
static int _countMatrices(PyArrayObject* w, int W, PyArrayObject* d, int D,
                          PyArrayObject* z, int T, PyArrayObject*** counts);

// 'expected' nw / nd count matrices (for soft/relaxed z-assignments)
static PyObject* expectedCountMatrices(PyObject* self, PyObject* args, 
                                       PyObject* keywds);

// Estimate phi and theta from count matrices (mean of posterior)
static PyObject* estPhiTheta(PyObject* self, PyObject* args, PyObject* keywds);

// MAP estimate of phi and theta from count matrices
static PyObject* mapPhiTheta(PyObject* self, PyObject* args, PyObject* keywds);
static int _mapPhiTheta(PyArrayObject* nw,  PyArrayObject* nd,
                        PyArrayObject* alpha, PyArrayObject* beta,
                        PyArrayObject*** phitheta);

// Calculate avg perplexity of (w,d) given (phi,theta)
static PyObject* perplexity(PyObject* self, PyObject* args, PyObject* keywds);

// Calculate LDA logike of (z,phi,theta) given (w,alpha,beta)
static PyObject* ldaLoglike(PyObject* self, PyObject* args, PyObject* keywds);
static double _ldaLoglike(PyArrayObject* w, PyArrayObject* d, PyArrayObject* z,
                          PyArrayObject* phi, PyArrayObject* theta,
                          PyArrayObject* alpha, PyArrayObject*beta);
static double _logDir(PyArrayObject* theta, PyArrayObject* alpha);
static double _logMult(PyArrayObject* counts, PyArrayObject* theta);

// Multinomial sampling function
static int mult_sample(double* vals, double sum);
