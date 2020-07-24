#include "Myligra1.h"
#include <cstdlib>

int Test(int IMG,int maxIters,int n,float* Bias,graph<asymmetricVertex>* GA){
	//GA.transpose();
	bool flag = false;
	float weight = 0.0625;
	float* Feature_cur = newA(float,n+1);
	float* Feature_next = newA(float,n+1);
	bool * frontier = newA(bool,n+1);
	memset(Feature_cur, 0, sizeof(float) * (n+1));
	memset(Feature_next, 0, sizeof(float) * (n+1));
	memset(frontier, 0, sizeof(bool) * (n+1));
	{for(int i=1;i<=image[IMG][0];++i){
		Feature_cur[image[IMG][i]] = 1;
		frontier[image[IMG][i]] = 1;
	}}
	vertexSubset Frontier(n+1,frontier);
	int iter = 1,t = 0;
	while(iter <= maxIters){
		//vertexSubset output = edgeMap(GA[iter],Frontier,Forward_F<asymmetricVertex>(Feature_cur,Feature_next));
		int m = Frontier.m;
		intT threshold = GA[iter].m/20;
		asymmetricVertex * G = GA[iter].V;
		if(m*33 > threshold){
			//Dense
			{for(int i=1;i<=n;++i){
				if(Frontier.d[i]){
					asymmetricVertex *vert = &G[i];
					uintE d = vert->getOutDegree();
					uintE * nghArray = vert->getOutNeighbors();
					for(int j = 0; j < d; j ++){
						uintE ngh = nghArray[j];
						float w = vert->weight[j];
						Feature_next[ngh] += Feature_cur[i] * w;
					}
				}
			}}
		}
		else{
			//Sparse
			Frontier.toSparse();
			asymmetricVertex *frontierVertices = newA(asymmetricVertex,m);
			for(int i=0;i < m;++i){
				asymmetricVertex v = G[Frontier.s[i]];
				frontierVertices[i] = v;
			}
			{for(int i = 0; i < m; i++){
				uintT v = Frontier.s[i];
				asymmetricVertex * vert = &frontierVertices[i];
				uintE d = vert->getOutDegree();
				uintE *nghArray = vert->getOutNeighbors();
				for(uintE j = 0; j < d; j++){
					uintE ngh = nghArray[j];
					float w = vert->weight[j];
					Feature_next[ngh] += Feature_cur[v] * w;
				}
			}}
			free(frontierVertices);
		}
		bool* next = newA(bool,n+1);
		for(int i=0;i<=n;++i)	next[i] = 1;
		for(int i=1;i<=n;++i){
			Feature_next[i] += Bias[i];
			if(Feature_next[i] <= 0){
				Feature_next[i] = 0;
				next[i] = 0;
			}
			else if(Feature_next[i] >= 32)
				Feature_next[i] = 32;
			Feature_cur[i] = 0;
		}			
		vertexSubset output(n+1,next);
		Frontier.del();
		Frontier = output;
		swap(Feature_cur,Feature_next);
		bool quit = true;
		for(int i=1;i<=n;++i) {
			if(Feature_cur[i] > 0) {
				quit = false;
				break;
			}
		}
		if(quit){
			flag = 1;
			break;
		}
		iter++;
	}
	free(Feature_next);
	free(Feature_cur);
	Frontier.del();
	if(flag)
		return 0;
	else
		return 1;
}
