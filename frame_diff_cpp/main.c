#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define w 2
#define h 2
#define d 3


void B_data_show(int *C,int size);
void A_data_show(int M[][][]);

int main(void)
{
	int A[w][h][d]={{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},sum=w*h*d;
	int *B = (int *)malloc(sizeof(int)*sum);
	memset(B,0,sizeof(int)*sum);
	int i,j,k;
	for(i=0;i<w;i++){
		for(j=0;j<h;j++){
			for(k=0;k<d;k++){
				printf("A:%d\n",*(*(*(A+i)+j)+k));
				*(B+i*h*d+j*d+k) = A[i][j][k];
				printf("B:%d\n",*(B+i*w+j*h+k));
			}
		}
	}
	// data_show(A,sum);
	B_data_show(B,sum);
//	A_data_show(A);
	return 0;
}
void B_data_show(int *C,int size)
{
	int i=0;
	for(i=0;i<size;i++){
		printf("C:%d\n",C[i]);
	}
}
/*
void A_data_show(int M[][][])
{
	int i,j,k;
	for(i=0;i<w;i++){
		for(j=0;j<h;j++){
			for(k=0;k<d;k++){
				printf("M:%d\n",M[i][j][k]);
			}
		}
	}
}
*/
