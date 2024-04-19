#include <cstdint>
#include <iostream>
#include <immintrin.h>

void plain_tmm(float *A, float *B, float *C,
               uint64_t M, uint64_t L, uint64_t N)
{
    for (uint64_t i=0; i<M; i++)
        for (uint16_t j=0; j<N; j++){
            float accum = 0.0f;
            for (uint64_t k = 0; k<L; k++)
                accum += A[i*L+k] * B[j*L+k];
            C[i*N+j] = accum;
        }
}


void avx2_tmm(float *A, float *B, float *C,
              uint64_t M, uint64_t L, uint64_t N)
{
    for (uint64_t i=0; i<M; i++)
        for (uint16_t j=0; j<N; j++){

            __m256 X = _mm256_setzero_ps();
            for (uint64_t k = 0; k<L; k+=8){
                const __m256 AV = _mm256_load_ps(A+i*L+k);
                const __m256 BV = _mm256_load_ps(B+j*L+k);
                X = _mm256_fmadd_ps(AV, BV, X);
            }
            C[i*N+j] = hsum_avx(X);
        }
}
