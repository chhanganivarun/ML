#include<stdio.h>
#include<time.h>

#include"knn_header.h"	//It has all the functions coressponding to the dataset.

int main()
{
	char filename[101];
	printf("Filename for the dataset:");
	strcpy(filename,md.filename);
	printf("%s\n",filename );

	FILE *datain = fopen(filename,"r");
	if ( datain == NULL )
	{
		printf("Error occurred while taking input from dataset.\nFile not found or doesn't have permissions\n");
		return 0;
	}

	long datasize;
	printf("Size of dataset:");
	datasize = md.datasize;
	printf("%ld\n",datasize );

	featureVector data[datasize];

	datasize = readData(datain,datasize,data);
	if(datasize<=0)
	{
		printf("Not Enough data\n");
		return 0;
	}
	shuffle(data,datasize,sizeof(data[0]));	//shuffle the input dataset

	float errorMat[Kmax][Pmax];
	long long k,p;

	int mink=0,minp=0;
	printf("kNN and r-fold cross validation started\n" );
	for(int i=1;i<=Kmax;i++)
	{
		for(int j=1;j<=Pmax;j++)
		{
			errorMat[i-1][j-1] = getError(datasize,data,i,j);	//get error for i as k and j as p
			printf("%f ", errorMat[i-1][j-1]);
			if(errorMat[i-1][j-1]<errorMat[mink][minp]) //Find minimum error
			{
				mink=i-1;
				minp=j-1;
			}
		}
		printf("\n" );
	}
	k = mink+1;
	p = minp+1;

	printf("Best value of k %lld and p %lld\nAccuracy:%f%%\n",k,p,100-errorMat[k-1][p-1]*100);

	fflush(datain);
	// fclose(datain);
}

