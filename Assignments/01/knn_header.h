#ifndef knn_header
#define knn_header

#include<stdio.h>
#include<stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "metadata.h"

#define Kmax 10
#define Pmax 10
#define R 5

typedef struct {
	float attributes[attrno];
	int class;
}featureVector;


long getMinClass(long datasize,double distance[datasize],featureVector data[datasize],int class,long k);
long readData(FILE *instream,long datasetSize,featureVector data[datasetSize]);


long readData(FILE *instream,long datasetSize,featureVector data[datasetSize])
{
	long i;
	size_t buffsize = 10000;
	char *line = (char *)malloc(sizeof(char)*buffsize);
	for(i=0;i<datasetSize && !feof(instream);i++)
	{
		getline(&line,&buffsize,instream);
		featureVector *fv;
		fv= malloc(sizeof(fv));
		const char space[4]="\t\n ,";
		char *token;

		token = strtok(line,space);
		int j;
		for(j=0;j<attrno && token;j++)
		{
			sscanf(token,"%f",&(fv->attributes[j]));
			token = strtok(NULL,space);
		}
		// printf("%d %d\n",j,attrno );
		sscanf(token,"%d",&(fv->class));

		data[i] = *fv;
		free(fv);
	}
	free(line);

	return i;
}

void printfv(featureVector fv)
{
	for(int i=0;i<attrno;i++)
	{
		printf("%f ",fv.attributes[i]);
	}
	printf("%d ",fv.class);
}

float distancefv(featureVector fv1,featureVector fv2, int p)
{
	double ret=0;
	for(int i=0;i<attrno;i++)
	{
		ret+=abs(pow(fv1.attributes[i]-fv2.attributes[i],p));
	}
	ret = pow(ret,1.0/p);
	return abs(ret);
}

void shuffle(void *array, size_t n, size_t size) {
    char tmp[size];
    char *arr = array;
    size_t stride = size * sizeof(char);
	srand(time(0));
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}


float getError(int datasize,featureVector data[datasize],long k,long p)
{
	//printf("For k %ld p %ld\n",k,p);
	double err=0;
	for(int i=0;i<R;i++)
	{
		int countpoints=0;
		double inerr=0;
		for(int j=0;j*R+i<datasize;j++,countpoints++)
		{
			double distance[datasize];
			for(int din=0;din<datasize;din++)
			{
				distance[din]=100;
			}
			//printf("k %d p %d partition %d entry %d\n",k,p,i,j*R+i );
			for(int pindex = 0 ; pindex < datasize ; pindex++)
			{
				//printf("distance for %d and %d\n",j*R+i,pindex);
				//printfv(data[j*R+i]);
				//printf(" and ");
				//printfv(data[pindex]);
				if(pindex>=i && !(pindex-i)%R)
				{
					distance[pindex]=2000000;
				}
				else
				{
					distance[pindex]=distancefv(data[j*R+i],data[pindex],p);
				}
				//printf(": %f \n",distance[pindex] );
			}
			for(int ssindex=0;ssindex*R+i<datasize;ssindex++)
			{
				distance[ssindex*R+i]=2000000;
			}
			/*
			for(int distind=0;distind<datasize;distind++)
			{
				printf("%f ",distance[distind]);
			}
			printf("\n");
			*/
			long cnt = getMinClass(datasize,distance,data,data[j*R+i].class,k);
			// printf("\ncnt %d\n",cnt );
			inerr+=((k-cnt*1.0)/k);

		}
		err+=inerr/countpoints;
	}
	return (err*1.0)/R;
}

long getMinClass(long datasize,double distance[datasize],featureVector data[datasize],int class,long k)
{
	long ret=0;
	for(long kindex=0;kindex<k;kindex++)
	{
		long min = 0;
		for(long i=0;i<datasize;i++)
		{
			if(distance[i]<distance[min])
			{
				min=i;
			}
		}
		if(data[min].class==class&& distance[min]!=2000000)
		{
			ret++;
		}
		distance[min]=2000000;
	}
	return ret;
}
#endif
