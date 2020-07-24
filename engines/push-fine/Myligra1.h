// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef LIGRA_H
#define LIGRA_H
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <algorithm>
#include "parallel.h"
#include "gettime.h"
#include "utils.h"
#include "vertex.h"
#include "compressedVertex.h"
#include "vertexSubset.h"
#include "graph.h"
#include "IO.h"
#include "parseCommandLine.h"
using namespace std;
int image[60005][65540];
bool truth_category[60005];
bool predict_category[60005];
//*****START FRAMEWORK*****

int Test(int IMG,int n,int maxIters,float* Bias,graph<asymmetricVertex> * GA);
long readimage(long n){
  printf("loading image\n");
  long cnt = 60000;
  int row,col,one;
  string file = "../../code/data/MNIST/sparse-images-"+to_string(n)+".tsv";
  FILE* fin2 = fopen(file.c_str(),"r");
  while(fscanf(fin2,"%d%d%d",&row,&col,&one) == 3){
    image[row][++image[row][0]] = col;
  }
  fclose(fin2);
  printf("image loaded->%ld\n",cnt);
  return cnt;
}
long readlayers(long n,long maxIters,graph<asymmetricVertex> * GA){
  long DNNedges = 0;
  float w = 0.0625;
  parallel_for(int iter = 1 ; iter <= maxIters; iter ++) {
      char a[25];
      long N = iter;
      int t = 0;
      while(N){
        a[t++] = N%10+48;
        N /= 10;
      }
      for(int i=0;i<t/2;++i)
        swap(a[i],a[t-1-i]);
      a[t] = '\0';
      string file = "../../code/data/DNN/neuron"+to_string(n)+"_Conversion/n"+to_string(n)+"-l"+to_string(iter);
      char c[55];
      memset(c,0,sizeof c);
      strcpy(c,file.c_str());
      GA[iter] = readGraph<asymmetricVertex>(c,0,0,0);
      GA[iter].transpose();
      asymmetricVertex* v = GA[iter].V;
      free(v[0].inNeighbors);
      for(int i=0;i<=n;++i)
        v[i].inNeighbors = NULL;
      v[0].weight = newA(float,GA[iter].m+1);
      for(int i=0;i<=GA[iter].m;++i){
        v[0].weight[i] = w;
      }
      for(int i=1;i<=n;++i){
        v[i].weight = v[i-1].weight+v[i-1].getOutDegree();
      }
      #pragma omp atomic
      DNNedges += GA[iter].m;
      
  }
  return DNNedges;
}
int readcategory(long n,long layer) {
  memset(truth_category, 0, sizeof(bool) * 60005);
  memset(predict_category, 0, sizeof(bool) * 60005);
  string file = "../../code/data/DNN/neuron"+to_string(n)+"-l"+to_string(layer)+"-categories.tsv";
  FILE* fp = fopen(file.c_str(), "r");
  int i, cnt = 0;
  while(fscanf(fp, "%d", &i) == 1) {
    truth_category[i] = true;
    cnt ++;
  }
  return cnt;
}

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv);
  int maxLayers = P.getOptionLongValue("-l",120);
  int neurons = P.getOptionLongValue("-n",1024);
  int threads = P.getOptionLongValue("-t",48);
  int t = 1024;
  cout << maxLayers << " " << neurons<< endl;
  float bias = -0.30;
  while(t != neurons){
      t *= 4;
      bias -= 0.05;
  }
  float* Bias = newA(float, neurons+1);
  parallel_for(int i=0;i<=neurons;i++) Bias[i] = bias;
  long cnt = readimage(neurons);
  graph<asymmetricVertex> *G = newA(graph<asymmetricVertex>,maxLayers+1);
  long DNNedges = readlayers(neurons,maxLayers,G);
  int truth_pictures = readcategory(neurons,maxLayers);
  
  float time_use=0;
  struct timeval start;
  struct timeval end;
  int wrong_number = 0, out_pictures = 0;
  gettimeofday(&start,NULL);
  parallel_for(int IMG=1;IMG<=cnt;++IMG){
      int ans = Test(IMG,maxLayers,neurons,Bias,G);
      if(ans) {
          if(truth_category[IMG] == false) {
            cout << "Image " << IMG << " Wrong output." << endl;
            wrong_number ++;
          }
          predict_category[IMG] = true;
      }
  }
  gettimeofday(&end,NULL);
  time_use=(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)*1e-6;
  cout << "DNN neurons/layers: " << neurons << ", Layers: " << maxLayers;
  cout << ", Edges: " << DNNedges << endl;
  cout << "runtime: " << time_use << "s" <<  endl;
  cout << "run rate(edges/sec):" << 1.0*cnt*DNNedges/time_use << endl;
  if(wrong_number || memcmp(truth_category, predict_category, sizeof(bool) * 60001)) {
    cout << "Total wrong number: " << wrong_number << endl;
    cout << "Output pictures: " << out_pictures << ", truth output pictures: " << truth_pictures << endl;
    cout << "Challenge Failed!\n";
  }
  else cout << "Challenge Passed!" << endl;
    cout << "total number of output pictures:"<<truth_pictures << endl;

  for(int i=1;i<=maxLayers;++i){
    free(G[i].V[0].weight);
    G[i].V[0].weight = NULL;
    G[i].del();
  }
  return 0;
}
#endif
