#include "Myligra1.h"
#include <cstdlib>

int Test(int IMG,int maxIters,int n,float* Bias,graph<asymmetricVertex>* GA){
	bool flag = false;
	float* Feature_cur = newA(float,n+1);
	float* Feature_next = newA(float,n+1);
	memset(Feature_cur, 0, sizeof(float) * (n+1));
	memset(Feature_next, 0, sizeof(float) * (n+1));
	{for(int i=1;i<=image[IMG][0];++i){
		Feature_cur[image[IMG][i]] = 1;
	}}
	int iter = 1,t = 0;
	while(iter <= maxIters){
		asymmetricVertex * G = GA[iter].V;
		{for(int i=1;i<=n;++i){
			asymmetricVertex *vert = &G[i];
			uintE d = vert->getInDegree();
			uintE * nghArray = vert->getInNeighbors();
			for(uintE j = 0; j < d; j ++){
				uintE ngh = nghArray[j];
				if(Feature_cur[ngh] == 0) continue;
				float w = vert->weight[j];
				Feature_next[i] += Feature_cur[ngh] * w;
			}
		}}
		for(int i=1;i<=n;++i){
			Feature_next[i] += Bias[i];
			if(Feature_next[i] <= 0){
				Feature_next[i] = 0;
			}
			else if(Feature_next[i] >= 32)
				Feature_next[i] = 32;
			Feature_cur[i] = 0;
		}
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
	if(flag)
		return 0;
	else
		return 1;
}
