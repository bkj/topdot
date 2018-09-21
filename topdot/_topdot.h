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

#ifndef UTILS_CPPCLASS_H
#define UTILS_CPPCLASS_H

extern void _topdot(
  int n_row, int n_col,
  int Ap[], int Aj[], double Ax[],
  int Bp[], int Bj[], double Bx[],
  int k, double lower_bound,
  int Cj[], double Cx[]
);

#endif //UTILS_CPPCLASS_H
