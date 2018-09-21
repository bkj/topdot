#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at#
#    http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Cool Kids in EM
# April 20, 2017

# distutils: language = c++

import numpy as np
cimport numpy as np

cdef extern from "_topdot.h":
    cdef void _topdot(
                        int n_row,
                        int n_col,
                        int Ap[],
                        int Aj[],
                        double Ax[],
                        int Bp[],
                        int Bj[],
                        double Bx[],
                        int topn,
                        double lower_bound,
                        int Cj[],
                        double Cx[]);


cpdef topdot(
        int n_row,
        int n_col,
        np.ndarray[int, ndim=1] a_indptr,
        np.ndarray[int, ndim=1] a_indices,
        np.ndarray[double, ndim=1] a_data,
        np.ndarray[int, ndim=1] b_indptr,
        np.ndarray[int, ndim=1] b_indices,
        np.ndarray[double, ndim=1] b_data,
        int k,
        double lower_bound,
        np.ndarray[int, ndim=1] c_indices,
        np.ndarray[double, ndim=1] c_data
    ):
    
    cdef int* Ap    = &a_indptr[0]
    cdef int* Aj    = &a_indices[0]
    cdef double* Ax = &a_data[0]
    cdef int* Bp    = &b_indptr[0]
    cdef int* Bj    = &b_indices[0]
    cdef double* Bx = &b_data[0]
    cdef int* Cj    = &c_indices[0]
    cdef double* Cx = &c_data[0]
    
    _topdot(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, k, lower_bound, Cj, Cx)
    return
