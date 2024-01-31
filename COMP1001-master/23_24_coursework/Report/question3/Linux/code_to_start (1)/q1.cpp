/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
__m128 num2, num3, num4, num5, num6, num7, num8;
__m128 loadp, loadq, loada, loadx, loadz, mulpAl, mulzBe, result, mulaAl, mulx, subqBe;
__m128 alpha_vector, beta_vector;
void initialize();
bool check();
void routine1(float alpha, float beta);
void routine1_vec(__m128 aplha_vector, __m128 beta_vector);
void routine2(float alpha, float beta);
void routine2_vec(__m128 aplha_vector, __m128 beta_vector);

__declspec(align(64)) float  y[M], z[M], p[M];
__declspec(align(64)) float A[N][N], x[N], w[N], q[N];
int main() {

    float alpha = 0.023f, beta = 0.045f; 
    bool difference = false;
    alpha_vector = _mm_set_ps(0.023f, 0.023f, 0.023f, 0.023f);
    beta_vector = _mm_set_ps(0.045f, 0.045f, 0.045f, 0.045f);
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer
    for (t = 0; t < TIMES1; t++)
    {
        routine1(alpha, beta);
    }

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));


    printf("\nRoutine1_vec:");
    start_time = omp_get_wtime(); //start timer
    for (t = 0; t < TIMES1; t++)
    {
        routine1_vec(alpha_vector, beta_vector);
    }
    printf("%f  %f", y[50], p[50]);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));
    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer
    for (t = 0; t < TIMES2; t++)
    {
        routine2(alpha, beta);
    }

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));


    printf("\nRoutine2_vec:");
    start_time = omp_get_wtime(); //start timer
    for (t = 0; t < TIMES1; t++)
    {
        routine2_vec(alpha_vector, beta_vector);
    }

    printf("%f  %f", w[50], q[50]);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    difference = check();
    printf("Were there any differences: ");
    printf("%s", difference ? "True" : "False");

    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine2 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
        q[i] = w[i];
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
        p[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
    {
         y[i] = alpha * y[i] + beta * z[i];
        
    }
}

void routine1_vec(__m128 alpha_vector, __m128 beta_vector)
{
    for (unsigned int i = 0; i < M; i+=4)
    {
        loadp = _mm_loadu_ps(&p[i]);
        loadz = _mm_loadu_ps(&z[i]);
        mulpAl = _mm_mul_ps(loadp, alpha_vector);
        mulzBe = _mm_mul_ps(loadz, beta_vector);
        result = _mm_add_ps(mulpAl, mulzBe);
        _mm_storeu_ps(&p[i],result);
        
    }
}

void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];
           
        }

}

void routine2_vec(__m128 alpha_vector, __m128 beta_vector)
{
    for (unsigned int i = 0; i < N; i++)
        for (unsigned int j = 0; j < N; j++)
        {
            loadq = _mm_loadu_ps(&q[i]);
            loada = _mm_loadu_ps(&A[i][j]);
            loadx = _mm_loadu_ps(&x[j]);
            mulaAl = _mm_mul_ps(alpha_vector, loada);
            mulx = _mm_mul_ps(mulaAl, loadx);
            subqBe = _mm_sub_ps(loadq, beta_vector);
            result = _mm_add_ps(subqBe, mulx);
            _mm_store_ss(&q[i], result);
            
        }
}

bool check()
{
    for(unsigned int i = 0; i < M; i++)
    {
        if (p[i] != y[i])
        {
            printf("difference found at position %d in routine 1\n", i);
            return true;
        }
    }

    for (unsigned int i = 0; i < N; i++)
    {
        if (q[i] != w[i])
        {
            printf("difference found at position %d in routine 2\n", i);
            return true;
        }
    }
    return false;
}




