#!/usr/bin/env python

"""
    test.py
"""

from __future__ import print_function, division

import sys
import json
import argparse
import numpy as np
from time import time
from scipy import sparse

import topdot.topdot as td

def run_topdot(A, B, k, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    
    num_rows = A.shape[0]
    
    I = np.empty(num_rows * k, dtype=np.int32)
    D = np.empty(num_rows * k, dtype=A.dtype)
    td.topdot(
        n_row=num_rows,
        n_col=B.shape[1],
        
        a_indptr=np.asarray(A.indptr, dtype=np.int32),
        a_indices=np.asarray(A.indices, dtype=np.int32),
        a_data=A.data,
        
        b_indptr=np.asarray(B.indptr, dtype=np.int32),
        b_indices=np.asarray(B.indices, dtype=np.int32),
        b_data=B.data,
        
        k=k,
        lower_bound=lower_bound,
        
        c_indices=I,
        c_data=D,
    )
    
    I = np.array(I).reshape(num_rows, k)
    D = np.array(D).reshape(num_rows, k)
    
    return D, I

def _run_naive(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return (None, None)
    elif nnz <= ntop:
        return csr_row.data, csr_row.indices
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        return csr_row.indices[arg_idx], csr_row.data[arg_idx]

def run_naive(A, B, ntop, lower_bound=None):
    C = A.dot(B)
    I, D = zip(*[_run_naive(row, ntop) for row in C])
    return D, I

# --
# Run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=4096)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    t = time()
    A = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    B = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    A *= 10
    B *= 10
    gen_time = time() - t
    print('gen_time  ', gen_time, file=sys.stderr)
    
    t = time()
    td_D, td_I = run_topdot(A, B, args.k)
    td_time = time() - t
    print('td_time   ', td_time, file=sys.stderr)
    
    t = time()
    na_D, na_I = run_naive(A, B, args.k)
    naive_time = time() - t
    print('naive_time', naive_time, file=sys.stderr)
    
    rand_idx = np.random.choice(args.dim, args.k, replace=False)
    for idx in rand_idx:
        na_idx = sorted(na_I[idx])
        td_idx = sorted(td_I[idx])
        assert (na_idx == td_idx), "(na_idx != td_idx)"
    
    t = time()
    cc = A.dot(B)
    dot_time = time() - t
    print('dot_time  ', dot_time, file=sys.stderr)
    
    print(json.dumps({
        "gen_time"   : gen_time,
        "td_time"    : td_time,
        "naive_time" : naive_time,
        "dot_time"   : dot_time,
    }))
    
    # # >>
    # # FAISS w/ dense matrices
    # import faiss
    # findex  = faiss.IndexFlatIP(args.dim)
    # Ad = np.asarray(A.todense()).astype(np.float32)
    # Bd = np.asarray(B.todense()).astype(np.float32)
    # Bd = np.ascontiguousarray(Bd.T)
    # findex.add(Bd)
    
    # t = time()
    # faiss_D, faiss_I = findex.search(Ad, args.k)
    # faiss_brute_time = time() - t
    # print('faiss_brute_time', faiss_brute_time)
    
    # rand_idx = np.random.choice(args.dim, args.k, replace=False)
    # for idx in rand_idx:
    #     faiss_idx = sorted(na_I[idx])
    #     td_idx    = sorted(td_I[idx])
    #     assert (faiss_idx == td_idx), "(faiss_idx != td_idx)"
    # # <<
    
    # # >>
    # NMSLIB w/ sparse matrices
    # ... slower than brute force? ...
    # # <<