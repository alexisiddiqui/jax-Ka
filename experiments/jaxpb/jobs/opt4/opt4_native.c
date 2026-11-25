/* Standalone OPT4 arm. No production dispatch. IEEE float; no fast-math. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
void opt4_sor(float *p, const float *rhs, const float *ex, const float *ey,
              const float *ez, const float *dv, const float *om, const int *op,
              int *its, float *mcs, int lanes, int n, int sweeps, int converge, int threads) {
 const long nn=(long)n*n*n;
 #pragma omp parallel for num_threads(threads) schedule(static)
 for(int l=0;l<lanes;l++) {
  float *q=p+l*nn; const float *r=rhs+l*nn;
  const float *x=ex+op[l]*nn,*y=ey+op[l]*nn,*z=ez+op[l]*nn,*d=dv+op[l]*nn;
  float omega=om[l],mc=1e9f;
  for(int s=1;s<=sweeps;s++) {
   float maxc=0.f;
   for(int c=0;c<2;c++) for(int i=1;i<n-1;i++) for(int j=1;j<n-1;j++) {
    int k=1; if(((i+j+k)&1)!=c) ++k;
    for(;k<n-1;k+=2) {
     long a=((long)i*n+j)*n+k;
     float nb=x[a]*q[a+n*n]+x[a-n*n]*q[a-n*n]+y[a]*q[a+n]+y[a-n]*q[a-n]+z[a]*q[a+1]+z[a-1]*q[a-1];
     float gs=(nb+r[a])*d[a],old=q[a];
     float val=old+omega*(gs-old); q[a]=val;
     if(s%10==0) { float change=fabsf(val-old); if(change>maxc) maxc=change; }
    }
   }
   if(s%10==0) { mc=maxc; if(converge && mc<.01f) {its[l]=s; break;} }
   its[l]=s;
  }
  mcs[l]=mc;
 }
}
