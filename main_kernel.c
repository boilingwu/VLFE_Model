//V2nd_mingus_test1 get familiar with the program 2017.08.14, on mingus
//VLFE 2nd stage test, for scatter model, investigating dynamics effect and static effect. 2017.08.10
//v8_test1
//modified at 16.10.18, computing kernel using symetric property.
//modified at 16.10.14, for plane fault validation with FEM by BAO
//test25.0-case4
//pack 2016.7.19
//  main.c
//  rupture_process_kernel
//

//  Copyright © 2016年 吴葆宁. All rights reserved.
//
//

//  main.c, oringinal main.cpp
//  rupture_processes
//  computing kernel
//  limit the slip in the x1 direction
//
//  finding a bug: for accurate computation of K(0,0,0,0,0)(=-mu/(2*Ct)),H(0) can't equal 0.5, it must be set as 0. 2016/5/1 morning
//
//  ATTENTION Global variebles in rupture_processes_main.cpp and rupture_processes.cpp must be exactly the same. Check it every time running this code.
//
// 4.22 23:51 change the discretization code, make it auto-suitable with LEN and WID.
//
// 5.11 finding bug at parallel writing, one core must start writing after the previous core close the file.
// transformed to c_version on 16/5/2
//read and write using binary 16/6/5
// debug for last slave_setting setting: the last core should take the rest elements. 16/6/5
// make the path become a pre-set parameter 2016.7.19
//  Created by 吴葆宁 on 16/4/19.
//  Copyright © 2016年 吴葆宁. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/*********************GENERAL PARAMETERS TO BE SET*************************/
#define LEN 100 //length(Element Num) in x1 direction
#define WID 100  //length(Element Num) in x2 direction
#define MAXTIME 300 //time duration
#define DELTA_S 0.1 //space interval
#define ct 3.23 //km/s
#define cl 5.6 //km/s
#define pi 3.1415926535898
#define CFL 0.45
double mu=3000*ct*ct; //density of granite is assumed to be 3000kg/m^3, unit of mu is Mpa.
double DELTA_T=CFL*DELTA_S/cl; //time interval/s
const int ElementSize = (LEN+1)*(WID+1);

//folder_path
char folder_path[255] = "/Users/Baoning/Dropbox/VLFE_2ndStage/v2nd_test4";
/*********************GENERAL PARAMETERS TO BE SET*************************/

//function
double sgn(double x);
double H(double x);
double fabs(double x);
double g1(double x1,double x2,double c,double t);
double g2(double x1,double x2,double c,double t);
double g3(double x1,double x2,double c,double t);
double L11(double x1,double x2,double t);
double L22(double x1,double x2,double t);
double L21(double x1,double x2,double t);
double L12(double x1,double x2,double t);

//struct initiation
struct Element{
    int len;
    int wid;
};

//initiation array

char path[255];




/*********************NOW IS THE MAIN FUNCTION*************************/
int main(int argc, const char * argv[]) {
    // insert code here...
    
    //timing
    clock_t start,finish;
    double local_start_mpi, local_finish_mpi, duration, local_elapsed, elapsed ;
    
    //discretization
    double temp_storage = 0;
    int i=0, j=0, l=0, m=0, tau=0, p=0, q=0, my_rank, comm_sz; //tau represent (k-n) in rupture simulation program
    
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    //initiate Element for non-structral kernel
    int count=-1;
    double x1[LEN+1];
    double x2[WID+1];
    double t[MAXTIME+1];
    struct Element E[(LEN+1)*(WID+1)];
    
    for(i=0;i<=LEN;i++){
        for(j=0;j<=WID;j++){
            count++;
            E[count].len = i;
            E[count].wid = j;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //setting real coodinates according to sequence number
    for(i=0;i<=LEN;i++){
        x1[i] = (-(double)LEN*0.5+(double)i) * DELTA_S;
    }
    for(j=0;j<=WID;j++){
        x2[j] = (-(double)WID*0.5+(double)j) * DELTA_S;
    }
    for(tau=0;tau<=MAXTIME;tau++){
        t[tau] = 0.0+(double)tau*DELTA_T;
    }
    
    double half_intervel = 0.5*DELTA_S;
    double time_step = DELTA_T;
    
    start = clock();
    
    
    /********************** Parellel part *************************/
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    local_start_mpi = MPI_Wtime();
    
    int local_count=-1;
    int local_ElementSize, local_bott, local_top, receive_signal=1;
    local_ElementSize = ElementSize/comm_sz;
    
    local_bott = my_rank*local_ElementSize;
    if(my_rank+1 == comm_sz){
        local_top = ElementSize - 1;
        local_ElementSize = local_top-local_bott+1;
    }else{
        local_top = (my_rank+1)*local_ElementSize-1;
    }
    printf("my_rank = %d, local_ElementSize = %d\n", my_rank, local_ElementSize);
    
    double *Kernel = malloc(sizeof(double)*(local_ElementSize*(MAXTIME+1)));
    
    for(p=local_bott;p<=local_top;p++){
        if(my_rank == 0){
            printf("now computing Element %d\n",p);
        }
        //source element alway be l=0,m=0, this still works by the merit of symetric
        for(tau=0;tau<=MAXTIME;tau++){
            
            i=E[p].len;
            j=E[p].wid;
            l=0;
            m=0;
            
            temp_storage =
            L11(x1[i]-x1[l]-half_intervel,x2[j]-x2[m]-half_intervel,t[tau]+time_step)
            - L11(x1[i]-x1[l]-half_intervel,x2[j]-x2[m]-half_intervel,t[tau])
            - L11(x1[i]-x1[l]-half_intervel,x2[j]-x2[m]+half_intervel,t[tau]+time_step)
            + L11(x1[i]-x1[l]-half_intervel,x2[j]-x2[m]+half_intervel,t[tau])
            - L11(x1[i]-x1[l]+half_intervel,x2[j]-x2[m]-half_intervel,t[tau]+time_step)
            + L11(x1[i]-x1[l]+half_intervel,x2[j]-x2[m]-half_intervel,t[tau])
            + L11(x1[i]-x1[l]+half_intervel,x2[j]-x2[m]+half_intervel,t[tau]+time_step)
            - L11(x1[i]-x1[l]+half_intervel,x2[j]-x2[m]+half_intervel,t[tau]);
            
            local_count++;//first number of local_count is 0
            Kernel[local_count] = temp_storage;
            
            
        }
    }
    
    FILE *fp;

    printf("my_rank=%d\n", my_rank);
    if(my_rank != 0){
        //waiting the previous core to finish writing.
        MPI_Recv(&receive_signal, 1, MPI_INT, my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 //       printf("core %d received signal\n", my_rank);
    }
    //二进制写入
    else{ //my_rank == 0
        sprintf(path,"%s/kernel.dat",folder_path);
        fp = fopen(path,"w"); //将原有文件长度截为0
        fclose(fp);
    }
    sprintf(path,"%s/kernel.dat",folder_path);
    fp = fopen(path,"a+"); //core[my_rank]打开文件，采用可添加的fopen，便于多核写入。
    if(fp) printf("open ok\n");
    
    int actual_writingnum = fwrite(Kernel,sizeof(double),local_ElementSize*(MAXTIME+1),fp);//writing
    if(actual_writingnum != local_ElementSize*(MAXTIME+1)) printf("ERROR!!!!\n");
    
    fclose(fp);
    printf("core %d finished writing\n", my_rank);
    if(my_rank != comm_sz-1){
        printf("core %d entering sending\n", my_rank);
        MPI_Send(&receive_signal, 1, MPI_INT, my_rank+1, 0, MPI_COMM_WORLD);
        printf("core %d sending receive_signal successfully\n", my_rank);
    }

    
    free(Kernel);
    MPI_Barrier(MPI_COMM_WORLD);
    
    finish = clock();
    duration = (double)(finish-start)/CLOCKS_PER_SEC;
    
    local_finish_mpi = MPI_Wtime();
    local_elapsed = (local_finish_mpi-local_start_mpi);
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(my_rank == 0){
        printf("using %lf sec (computed by mpi.h)\n",elapsed);
        printf("using %lf sec (computed by time.h)\n",duration);
    }
    MPI_Finalize();
   /******************** End of Parallel ***************************/
    
    return 0;
    
}

//绝对值函数
double fabs(double x){
    
    double answer = 0.0;
    
    if(x < 0){
        answer = -x;
    }
    else if(x > 0){
        answer = x;
    }
    else {
        answer = 0.0;
    }
    return answer;
}

// sgn符号函数
double sgn(double x){
    
    double answer = 0.0;
    
    if(x < 0){
        answer = -1.0;
    }
    else if(x > 0){
        answer = 1.0;
    }
    else {
        answer = 0.0;
    }
    return answer;
}

//Heaviside函数，0取0.5
double H(double x){
    
    double answer = 0.0;
    
    if(x < 0){
        answer = 0.0;
    }
    else if(x > 0){
        answer = 1.0;
    }
    else {
        answer = 0.0;
    }
    return answer;
}

//g1 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS
double g1(double x1,double x2,double c,double t){
    
    double answer = 0.0;
    double xr = 0.0;
    
    xr = sqrt(x1*x1+x2*x2);
    
    
    if((t*t-(x1/c)*(x1/c)) < 0){
        answer = H(t-xr/c) * ((x2*(2.0*xr*xr+x1*x1))*pow(t,3.0)/(2.0*pow(xr,3.0))-(3.0*x1*x1*x2*t)/(2.0*c*c*xr));
    }
    else{
        answer = pow((t*t-(x1/c)*(x1/c)),3.0/2.0) * (2.0*H(x2)*H(t-fabs(x1)/c)-sgn(x2)*H(t-xr/c)) + H(t-xr/c) * ((x2*(2.0*xr*xr+x1*x1))*pow(t,3.0)/(2.0*pow(xr,3.0))-(3.0*x1*x1*x2*t)/(2.0*c*c*xr));
    }
    return answer;
}

//g2 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS
double g2(double x1,double x2,double c,double t){
    
    double answer = 0.0;
    double xr = 0.0;
    
    xr = sqrt(x1*x1+x2*x2);
    
    
    if((t*t-(x1/c)*(x1/c)) < 0){
        answer = H(t-xr/c)*(x2*t)/xr;
    }
    else{
        answer = pow((t*t-(x1/c)*(x1/c)),1.0/2.0) * (2.0*H(x2)*H(t-fabs(x1)/c)-sgn(x2)*H(t-xr/c)) + H(t-xr/c)*(x2*t)/xr;
    }
    return answer;
}

//g3 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS
double g3(double x1,double x2,double c,double t){
    
    double answer = 0.0;
    double xr = 0.0;
    
    xr = sqrt(x1*x1+x2*x2);
    
    
    if((t*t-(x1/c)*(x1/c)) < 0){
        answer = atan(H(t-xr/c)*x2/x1);
    }
    else{
        answer = atan(pow((t*t-(x1/c)*(x1/c)),1.0/2.0)/(x1/c)) * (2.0*H(x2)*H(t-fabs(x1)/c)-sgn(x2)*H(t-xr/c)) + atan(H(t-xr/c)*x2/x1);
    }
    return answer;
}

//L11 function in page 246 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS
double L11(double x1,double x2,double t){
    
    double answer = 0;
    
    answer = -mu/(2.0*ct)*H(x1)*H(x2)*H(t) - (mu*ct*ct)/(3.0*pi*pow(x1,3.0))*(g1(x1,x2,cl,t)-g1(x1,x2,ct,t)) - mu/(4.0*pi*x2)*g2(x2,x1,ct,t) + mu/(4.0*pi*ct)*(g3(x1,x2,ct,t)+g3(x2,x1,ct,t));
    
    return answer;
}

//L22 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS, symatric with L11
double L22(double x1,double x2,double t){
    
    double answer = 0;
    
    answer = -mu/(2.0*ct)*H(x2)*H(x1)*H(t) - (mu*ct*ct)/(3.0*pi*pow(x2,3.0))*(g1(x2,x1,cl,t)-g1(x2,x1,ct,t)) - mu/(4.0*pi*x1)*g2(x1,x2,ct,t) + mu/(4.0*pi*ct)*(g3(x2,x1,ct,t)+g3(x1,x2,ct,t));
    
    return answer;
}

//L21 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS
double L21(double x1,double x2,double t){
    
    double answer = 0;
    double xr = sqrt(x1*x1+x2*x2);
    
    answer = (mu*ct*ct)/(6.0*pi*pow(xr,3.0)) * (pow(t,3.0)-(3.0*xr*xr*t)/(cl*cl)+(2.0*pow(xr,3.0))/pow(cl,3.0)) * H(t-xr/cl) - (mu*ct*ct)/(6.0*pi*pow(xr,3.0)) * (pow(t,3.0)-(3.0*xr*xr*t)/(ct*ct)+(2.0*pow(xr,3.0))/(2.0*pow(ct,3.0))) * H(t-xr/ct);
    
    return answer;
}

//L12 function in page 247 FAULT ZONE PEOPERTIES AND EARTHQUAKE RUPTURE DYNAMICS, equal to L11
double L12(double x1,double x2,double t){
    
    double answer = 0;
    double xr = sqrt(x1*x1+x2*x2);
    
    answer = (mu*ct*ct)/(6.0*pi*pow(xr,3.0)) * (pow(t,3.0)-(3.0*xr*xr*t)/(cl*cl)+(2.0*pow(xr,3.0))/pow(cl,3.0)) * H(t-xr/cl) - (mu*ct*ct)/(6.0*pi*pow(xr,3.0)) * (pow(t,3.0)-(3.0*xr*xr*t)/(ct*ct)+(2.0*pow(xr,3.0))/(2.0*pow(ct,3.0))) * H(t-xr/ct);
    
    return answer;
}

//generalize the L(alpha)/(beta) function for further use
double L(int alpha,int beta,double x1,double x2,double t){
    
    double answer = 0;
    
    if(alpha == 1 && beta == 1){
        answer = L11(x1,x2,t);
    }
    else if(alpha == 1 && beta == 2){
        answer = L12(x1,x2,t);
    }
    else if(alpha == 2 && beta == 1){
        answer = L21(x1,x2,t);
    }
    else if(alpha == 2 && beta == 2){
        answer = L22(x1,x2,t);
    }
    else{
        answer = 0;
    }
    return answer;
}
