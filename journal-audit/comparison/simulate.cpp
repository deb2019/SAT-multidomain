// Three loss strata: changed positives, changed negatives, and agreements.
// Inputs: N D K direction(+1=lower SAT threshold). Reuse already queried labels.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
int main(){
 std::mt19937_64 rng(20261126);
 std::uniform_real_distribution<double> uniform;
 int n,d,k,sign;
 const int repetitions=4000;
 std::cout<<std::setprecision(13);
 while(std::cin>>n>>d>>k>>sign){
   double delta=2.0*d/n;
   double wd=std::sqrt((std::pow(5-delta,2)+std::pow(-1-delta,2))/2);
   double wa=delta,total=d*wd+(n-d)*wa;
   std::array<int,3> caps={10,25,50};
   double sums[3][11]={};
   if(d>0){
    for(int rep=0;rep<repetitions;++rep){
     int seenP=0,seenM=0,seenA=0,drawP=0,drawM=0,drawA=0,unique=0;
     for(int j=0;j<3;j++){
      while(unique<std::min(caps[j],n)){
       double u=uniform(rng)*total;
       if(u<k*wd){++drawP;if(u>=(seenP*wd)){++seenP;++unique;}}
       else if(u<d*wd){++drawM;if(u>=(k+seenM)*wd){++seenM;++unique;}}
       else {++drawA;if(u>=d*wd+seenA*wa){++seenA;++unique;}}
      }
      bool deploy=sign*(5*drawP-drawM)>0;
      double remaining=sign*(6*k-d-(5*seenP-seenM));
      // Self-normalised importance weights; their common normalisation cancels.
      double rel=wd/wa;
      double wsum=drawP+drawM+rel*drawA;
      double estimate=sign*(5*drawP-drawM)/wsum;
      double se=std::sqrt(drawP*std::pow(5*sign-estimate,2)+drawM*std::pow(-sign-estimate,2)+drawA*std::pow(rel*estimate,2))/wsum;
      bool wald=se>1e-12 && estimate>1.6448536269514722*se;
      bool waldLimit=wald || (se<=1e-12 && estimate>0);
      sums[j][8]+=waldLimit;sums[j][9]+=waldLimit*remaining;sums[j][10]+=waldLimit*remaining*remaining;
      sums[j][5]+=wald;sums[j][6]+=wald*remaining;sums[j][7]+=wald*remaining*remaining;
      sums[j][0]+=deploy;sums[j][1]+=deploy*remaining;sums[j][2]+=unique;
      sums[j][3]+=deploy*remaining*remaining;sums[j][4]+=seenP+seenM;
     }
    }
   }
   for(int j=0;j<3;j++)for(double x:sums[j])std::cout<<x/repetitions<<' ';
   std::cout<<'\n';
 }
}
