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

// Do a single IN-PLACE Gibbs sample for standard LDA
static PyObject* standardGibbs(PyObject *self, PyObject *args, PyObject* keywds);

// Do a single IN-PLACE Gibbs sample for z-label LDA
static PyObject * zLabelGibbs(PyObject *self, PyObject *args, PyObject* keywds);

// Online construction of an initial z-sample
static PyObject* onlineInit(PyObject *self, PyObject *args, PyObject* keywds);

// Build nw and nd count matrices
static PyObject * countMatrices(PyObject *self, PyObject *args, PyObject* keywds);

// Estimate phi and theta from count matrices
static PyObject * estPhiTheta(PyObject *self, PyObject *args, PyObject* keywds);

// Calculate avg perplexity of (w,d) given (phi,theta)
static PyObject * perplexity(PyObject *self, PyObject *args, PyObject* keywds);

// Multinomial sampling function
static int mult_sample(double* vals, double sum);
