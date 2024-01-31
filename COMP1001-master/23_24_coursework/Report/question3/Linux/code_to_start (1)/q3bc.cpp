/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include "pgm.h"

//function declarations
void Gaussian_Blur();
void Sobel();
void SobelIntrinsics();
int initialize_kernel();
void read_image(const char *filename);
void read_image_and_put_zeros_around(char* filename);
void write_image2(const char* filename, unsigned char* output_image);
void write_image3(const char* filename, unsigned char* output_image);
void openfile(const char* filename, FILE** finput);
int getint(FILE* fp);

//CRITICAL POINT: images' paths - You need to change these paths
#define IN "C:/Users/theob/source/repos/1001 report question 3/input_images/a"
#define OUT "C:/Users/theob/source/repos/1001 report question 3/output_images/blurred"
#define OUT2 "C:/Users/theob/source/repos/1001 report question 3/output_images/edge detection"
#define OUT3 "C:/Users/theob/source/repos/1001 report question 3/output_images/edge detection intrinsics"

//IMAGE DIMENSIONS
int M = 400;  //cols
int N = 400; //rows
__m128i loadGxMask, loadGyMask, loadFilt;
__m128i mulGx, mulGy, addGyGx;

//CRITICAL POINT:these arrays are defined statically. Consider creating these arrays dynamically instead.
unsigned char *frame1;//input image

unsigned char *filt;//output filtered image
unsigned char* gradientIntrinsics;
unsigned char *gradient;//output image


const signed char Mask[5][5] = {//2d gaussian mask with integers
	{2,4,5,4,2} ,
	{4,9,12,9,4},
	{5,12,15,12,5},
	{4,9,12,9,4},
	{2,4,5,4,2}
};

const signed char GxMask[3][3] = {
	{-1,0,1} ,
	{-2,0,2},
	{-1,0,1}
};

const signed char GyMask[3][3] = {
	{-1,-2,-1} ,
	{0,0,0},
	{1,2,1}
};

char header[100];
errno_t err;

int main() {

	char* filePathIN; 
	filePathIN = (char*)malloc((sizeof(IN)+10) * sizeof(char));

	char* filePathOUT;
	filePathOUT = (char*)malloc((sizeof(OUT)+10) * sizeof(char));
	char* filePathOUT2; 
	filePathOUT2 = (char*)malloc((sizeof(OUT2)+10) * sizeof(char));
	char* filePathOUT3;
	filePathOUT3 = (char*)malloc((sizeof(OUT3) + 10) * sizeof(char));
	
	for (int i = 0; i < 31; i++)
	{
		sprintf(filePathIN, "%s%d.pgm", IN, i);
		#undef IN
		#define IN filePathIN
		
		read_image(IN);//read image from disc



		Gaussian_Blur(); //blur the image (reduce noise)
		Sobel(); //apply edge detection
		SobelIntrinsics();
		sprintf(filePathOUT, "%s%d.pgm", OUT, i);
		#undef OUT
		#define OUT filePathOUT

		sprintf(filePathOUT2, "%s%d.pgm", OUT2, i);
		#undef OUT2
		#define OUT2 filePathOUT2

		sprintf(filePathOUT3, "%s%d.pgm", OUT3, i);
		#undef OUT3
		#define OUT3 filePathOUT3

		write_image2(OUT, filt); //store output image to the disc
		write_image2(OUT2, gradient); //store output image to the disc
		write_image2(OUT3, gradientIntrinsics);
	}
	return 0;
}





void Gaussian_Blur() {

	int row, col, rowOffset, colOffset;
	int newPixel;
	unsigned char pix;
	//const unsigned short int size=filter_size/2;
	const unsigned short int size = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 0; row < N; row++) {
		for (col = 0; col < M; col++) {
			newPixel = 0;
			for (rowOffset = -size; rowOffset <= size; rowOffset++) {
				for (colOffset = -size; colOffset <= size; colOffset++) {

					if ((row + rowOffset < 0) || (row + rowOffset >= N) || (col + colOffset < 0) || (col + colOffset >= M))
						pix = 0;
					else
						pix = frame1[M * (row + rowOffset) + col + colOffset];

					newPixel += pix * Mask[size + rowOffset][size + colOffset];

				}
			}
			filt[M * row + col] = (unsigned char)(newPixel / 159);
		}
	}
}


void Sobel() {

	int row, col, rowOffset, colOffset;
	int Gx, Gy;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					Gx += filt[M * (row + rowOffset) + col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[M * (row + rowOffset) + col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			//gradient[M * row + col] = (unsigned char)sqrt(Gx * Gx + Gy * Gy); /* Calculate gradient strength		*/
			gradient[M*row + col] = abs(Gx) + abs(Gy); // this is an optimized version of the above

		}
	}


}

void SobelIntrinsics()
{

	int row, col, rowOffset, colOffset;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			__m128i Gx = _mm_set_epi32(0, 0, 0, 0);
			__m128i Gy = _mm_set_epi32(0, 0, 0, 0);


			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset+=8) {
				for (colOffset = -1; colOffset <= 1; colOffset+=8) {

					loadGxMask = _mm_loadu_si128((__m128i*)&GxMask[rowOffset + 1][colOffset + 1]);
					loadGyMask = _mm_loadu_si128((__m128i*)&GyMask[rowOffset + 1][colOffset + 1]);

					

					loadFilt = _mm_loadu_si128((__m128i*)(&filt[M * (row + rowOffset) + col + colOffset]));
					

					mulGx = _mm_maddubs_epi16(loadGxMask, loadFilt);



					mulGy = _mm_maddubs_epi16(loadGyMask, loadFilt);

					Gx = _mm_add_epi16(mulGx, Gx);

					Gy = _mm_add_epi16(mulGy, Gy);


				}
			}

			Gx = _mm_abs_epi8(Gx);
			Gy = _mm_abs_epi8(Gy);
			addGyGx = _mm_add_epi16(Gx, Gy);
			_mm_storeu_si128((__m128i*)&gradientIntrinsics[M * row + col], addGyGx);

		}
	}
}




void read_image(const char* filename)
{

	int c;
	FILE* finput;
	int i, j, temp;

	printf("\nReading %s image from disk ...", filename);
	openfile(filename, &finput);
	frame1 = (unsigned char*)realloc(frame1, N * M * sizeof(int));
	filt = (unsigned char*)realloc(filt, N * M * sizeof(int));
	gradientIntrinsics = (unsigned char*)realloc(gradientIntrinsics, N * M * sizeof(int));
	gradient = (unsigned char*)realloc(gradient, N * M * sizeof(int));

	if ((header[0] == 'P') && (header[1] == '5')) { //if P5 image

		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				//if (fscanf_s(finput, "%d", &temp,20) == EOF)
				//	exit(EXIT_FAILURE);
				temp = getc(finput);
				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else if ((header[0] == 'P') && (header[1] == '2')) { //if P2 image
		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				if (fscanf_s(finput, "%d", &temp) == EOF)
				{

					exit(EXIT_FAILURE);
				}
				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else {
		printf("\nproblem with reading the image");
		exit(EXIT_FAILURE);
	}

	fclose(finput);
	printf("\nimage successfully read from disc\n");

}



void write_image2(const char* filename, unsigned char* output_image)
{

	FILE* foutput;
	int i, j;



	printf("  Writing result to disk ...\n");

	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for writing\n", filename);
		perror("error");
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", output_image[M * j + i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}




void openfile(const char* filename, FILE** finput)
{
	int x0, y0, x, aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for reading\n", filename);
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	x0 = getint(*finput);//this is M
	y0 = getint(*finput);//this is N

	printf("\t header is %s, while x=%d,y=%d", header, x0, y0);


	//CRITICAL POINT: AT THIS POINT YOU CAN ASSIGN x0,y0 to M,N 

	M = x0;

	N = y0;
	printf("\n Image dimensions are M=%d,N=%d",M,N);


	x = getint(*finput); /* read and throw away the range info */
	//printf("\n range info is %d",x);

}



//CRITICAL POINT: you can define your routines here that create the arrays dynamically; now, the arrays are defined statically.



int getint(FILE* fp) /* adapted from "xv" source code */
{
	int c, i, firstchar;//, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], * sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
	   // if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',')
		//	garbage=1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c < '0' || c>'9') break;
	}
	return i;
}







