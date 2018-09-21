from time import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
import topdot.topdot as ct

def run(A, B, k, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    
    num_rows = A.shape[0]
    
    I = np.empty(num_rows * k, dtype=np.int32)
    D = np.empty(num_rows * k, dtype=A.dtype)
    ct.topdot(
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

def _slow_version(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return None
    elif nnz <= ntop:
        result = zip(gt_csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])
    
    return list(result)

def slow_version(A, B, ntop, lower_bound=0):
    C = A.dot(B)
    return [_slow_version(row, ntop) for row in C]

np.random.seed(123)

N = 14
a = rand(2 ** N, 2 ** N, density=0.01, format='csr')
b = rand(2 ** N, 2 ** N, density=0.01, format='csr')

t = time()
D, I = run(a, b, 1000)
print(time() - t)


t = time()
cc = slow_version(a, b, 1000)
print(time() - t)


for i in range(2 ** N):
    assert sorted([ccc[0] for ccc in cc[i]]) == sorted(I[i])


t = time()
cc = a.dot(b)
print(time() - t)


