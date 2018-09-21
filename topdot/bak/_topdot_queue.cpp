/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>
#include <queue>
#include <omp.h>

#include "./sparse_dot_topn_source.h"

struct candidate {int index; double value;};

class Compare
{
public:
    bool operator()(candidate c_i, candidate c_j) {
        return (c_i.value < c_j.value);
    }
};

void sparse_dot_topn_source(int n_row,
                        int n_col,
                        int Ap[],
                        int Aj[],
                        double Ax[], //data of A
                        int Bp[],
                        int Bj[],
                        double Bx[], //data of B
                        int ntop,
                        double lower_bound,
                        int Cj[],
                        double Cx[])
{
    
    #pragma omp parallel for
    for(int i = 0; i < n_row; i++){
        std::vector<int>    next(n_col, -1);
        std::vector<double> sums(n_col,  0);        
        std::priority_queue<candidate, std::vector<candidate>, Compare> pq;
        
        int head   = -2;
        int length =  0;

        int jj_start = Ap[i];
        int jj_end   = Ap[i+1];
        for(int jj = jj_start; jj < jj_end; jj++){
            int j = Aj[jj];
            double v = Ax[jj]; //value of A in (i,j)

            int kk_start = Bp[j];
            int kk_end   = Bp[j+1];
            for(int kk = kk_start; kk < kk_end; kk++){
                int k = Bj[kk]; //kth column of B in row j
                
                sums[k] += v * Bx[kk]; //multiply with value of B in (j,k) and accumulate to the result for kth column of row i
                
                if(next[k] == -1){
                    next[k] = head; //keep a linked list, every element points to the next column index
                    head    = k;
                    length++;
                }
            }
        }

        for(int jj = 0; jj < length; jj++){ //length = number of columns set (may include 0s)

            if(sums[head] > lower_bound){ //append the nonzero elements
                candidate c;
                c.index = head;
                c.value = sums[head];
                pq.push(c);
            }

            int temp = head;
            head = next[head]; //iterate over columns

            next[temp] = -1; //clear arrays
            sums[temp] =  0; //clear arrays
        }

        int len = std::min(ntop, (int)pq.size());

        for(int a = 0; a < len; a++){
            candidate c = pq.top();
            pq.pop();
            Cj[i * ntop + a] = c.index;
            Cx[i * ntop + a] = c.value;
        }
    }
    return;
}
