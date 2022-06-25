//V2nd_mingus_test3 two patch interaction
//VLFE 2nd stage test, for scatter model, investigating dynamics effect and static effect. 2017.08.10
//v8_test1
//mistake:wrong setting of back ground strenght
//bug:rate<0, forget to set localslipp before, now make it the value of last time step 2017.03.02
//bug:D>D0, local_slipp = sum2+rate[k][p]*DELTA_T; --->  local_slipp = sum2+local_ratekp*DELTA_T; 2017.03.02
//bug:line378 local_stress should equal Tres instead of 0; 2017.03.02
//adding Tres to enable that residue stress could be non-zero. 2017.02.22
//modified at 16.10.18, filter rate at every timestep (didn't filter slip and stress)
//modified at 16.10.18, computing kernel using symetric property.
// use MPI_FILE_ thing to open and read 2016.10.16，---》It works!!
//test25.0-case4
//  main.c, transformed from main.cpp
//  rupture_processes_main
//  limit the slip in the x1 direction
//
//  ATTENTION Global variebles in rupture_processes_main.cpp and rupture_processes.cpp must be exactly the same. Check it every time running this code.
//
// 4.22 23:51 change the discretization code, make it auto-suitable with LEN and WID.
//
// 2016.5.1 11:02 finding 2 bug, sum2*DELTA_T be wrongly write into sum2*DELTA_S; nucleation area setting: R ---> R^2
//transformed from main.c at 16/5/2.
//change the recv array size every time 0 recv one
//debug record: error didn't come from the wrong receiving number
//so far, good when ElementSize = n*comm_sz. 2016.5.14 noon
//read and write using binary 16/6/5
// debug for last slave_setting setting: the last core should take the rest elements. 16/6/5
//  Created by 吴葆宁 on 16/4/20.
//  Copyright © 2016年 吴葆宁. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/*********************GENERAL PARAMETERS TO BE SET*************************/
//Global variables, use granite as example
//share with kernel computing
#define LEN 100 //length(Element Num) in x1 direction
#define WID 100  //length(Element Num) in x2 direction
#define MAXTIME 300 //time duration
#define DELTA_S 0.1 //space interval
#define ct 3.23 //km/s
#define cl 5.6 //km/s
#define pi 3.1415926535898
#define CFL 0.45
#define StaticStressRadius 35//the radius in kernel that has significant static stress
#define NumAsperity1 2 //Number of Asperity in 1 direction
#define NumAsperity2 1 //Number of Asperity in 2 direction
double mu=3000*ct*ct; //density of granite is assumed to be 3000kg/m^3, unit of mu is Mpa.
double DELTA_T=CFL*DELTA_S/cl; //time interval/s
const int ElementSize = (LEN+1)*(WID+1);

double weigh_self=0.5, weigh_side=0.25, weigh_point=0.25; // filter's weigh parameter

double fabs(double x);

//folder_path
char folder_path[255] = "/Users/Baoning/Dropbox/VLFE_2ndStage/v2nd_test4";
/*********************GENERAL PARAMETERS TO BE SET*************************/

//function
double atof(const char *str);

//struct initiation
struct Element{
    int len;
    int wid;
};

//initiation array
double slip[(LEN+1)*(WID+1)];
double stress[(LEN+1)*(WID+1)];
double stress0[(LEN+1)*(WID+1)];
double Tu[(LEN+1)*(WID+1)];
//to change weakening curve while simulating, need to have a copy of Tu[p], name as Tu_0, 2017.03.11
double Tu_0[(LEN+1)*(WID+1)];
double rate[MAXTIME+1][(LEN+1)*(WID+1)];
double ratek[(LEN+1)*(WID+1)];//for parallel gathering at Master core(0 core)
double stress_all[MAXTIME+1][(LEN+1)*(WID+1)];
double slip_all[MAXTIME+1][(LEN+1)*(WID+1)];
int flag[(LEN+1)*(WID+1)];
int flag_1stepbefore[(LEN+1)*(WID+1)];
int flag_proc[MAXTIME+1][(LEN+1)*(WID+1)];
struct Element E[(LEN+1)*(WID+1)];
double ratek_filter[(LEN+1)*(WID+1)];//for filtering
char path[255];
int init_stress_flag[(LEN+1)*(WID+1)];
double patch_offset_1;
double patch_offset_2;
double eta, Tres;
double offset_1[NumAsperity1];
double offset_2[NumAsperity2];



/*********************NOW IS THE MAIN FUNCTION*************************/
int main(int argc, const char * argv[]) {
    
    // insert code here...
    //initiation
 /*********************FAULT AND FRICTION PARAMETERS TO BE SET*************************/
    //use only in this program
    int max_s_influence_timestep = 7;//maximum influence time duration(unit:step) after S wave arriving (change with kernel)
    double Tu0=0.01;//unit of stress, for convenience 假设震源在10km深，摩擦系数0.5，算出Tu大概在10^2Mpa,此处取50Mpa
    double UnitofD=2.0*Tu0*DELTA_S/mu; // unit of slip, for convinence
    //double UnitofV=2.0*Tu0*ct/mu; // unit of rate, for convinence
    double Ti=0.95*Tu0;//令Tu=1,Ti=1.2Tu,Te=0.6Tu
    double Te=0.0*Tu0;
    double Tres_0=0.0*Tu0;
    double eta_back=65000.0;
    double eta_patch=65000.0;
    double Aw=0.051*Tu0;
    double Lw=1*DELTA_S;
    double X0=-50*DELTA_S;
    //double D0=0.000000012; // critical distance (12cm)
    double HalfL_asperity=10*DELTA_S;//0.5 side length of asperity square
    //double R_asperity=4.9*DELTA_S;
    double offset_1 = -50*DELTA_S+HalfL_asperity;
    double offset_2 = 0*DELTA_S;
    double L_gap = 0.3*(2*HalfL_asperity);//the gap length between two patches, scale by patch side length
    double offset_triggering_1 = offset_1+L_gap+2*HalfL_asperity;
    //double R_barrier=50.0*DELTA_S;
 /*********************FAULT AND FRICTION PARAMETERS TO BE SET*************************/
    
    //program start
    int my_rank, comm_sz;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Status status;
    
    //timing
    clock_t start,finish;
    double local_start_mpi, local_finish_mpi, local_start_all, local_finish_all, duration, local_elapsed, elapsed;
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("now core %d start\n", my_rank);//timing
    local_start_all = MPI_Wtime();
    
    //discretization
    double x1[LEN+1];
    double x2[WID+1];
    double t[MAXTIME+1];
    int i=0, j=0, k=0, l=0, m=0, n=0, tau=0, p=0, q=0;
    
    for(i=0;i<=LEN;i++){
        x1[i] = (-(double)LEN*0.5+(double)i) * DELTA_S;
    }
    for(j=0;j<=WID;j++){
        x2[j] = (-(double)WID*0.5+(double)j) * DELTA_S;
    }
    for(k=0;k<=MAXTIME;k++){
        t[k] = 0.0+(double)k*DELTA_T;
    }
    
    //initiate Element for non-structral kernel
    int count=-1;
    
    for(i=0;i<=LEN;i++){
        for(j=0;j<=WID;j++){
            count++;
            E[count].len = i;
            E[count].wid = j;
        }
    }
    
    //set stress tag for elements
    int stress_flag_temp = 0;
    for(p=0;p<ElementSize;p++){
        stress_flag_temp = 0;
        i = E[p].len;
        j = E[p].wid;
        for(m=0;m<NumAsperity1;m++){ //NumAsperity1==2
            if(m==0){
                patch_offset_1 = offset_1;
            }else{//m==1
                patch_offset_1 = offset_triggering_1;
            }
            patch_offset_2 = offset_2;
            if(fabs(x1[i]-patch_offset_1)<=HalfL_asperity && fabs(x2[j]-patch_offset_2)<=HalfL_asperity){
                stress_flag_temp = 1;
            }
        }
        if(stress_flag_temp==1){
            init_stress_flag[p]=1;
        }
        else{//stress_flag_temp==0
            init_stress_flag[p]=0;
        }
    }
    
    for(p=0;p<ElementSize;p++){
        
        i = E[p].len;
        j = E[p].wid;
            
        slip[p] = 0;
        //set up the region of asperity, circle
        if(init_stress_flag[p]==1){
            if(fabs(x1[i]-X0)<Lw/2){
                stress0[p] = Ti + Aw*(1+cos(pi*(x1[i]-X0))/2);
            }
            else{
                stress0[p] = Ti;
            }
        }
        else{//init_stress_flag[p]==0
            if(fabs(x1[i]-X0)<Lw/2){
                stress0[p] = Tres_0 + Aw*(1+cos(pi*(x1[i]-X0))/2);
            }
            else{
                stress0[p] = Tres_0;
            }
        }
        stress[p] = stress0[p];
        stress_all[0][p] = stress[p];
//        if(i<5 || (LEN-i)<5 || j<5 || (WID-j)<5){
        if(0 != 0){
            Tu[p] = 10.0*Tu0;
        }
        else{
            if(init_stress_flag[p]==0){
                //Tu[p] = 1.0*Tu0;//backgroud, strenght equal peak strength.
                Tu[p] = Tres_0;//backgroud, strenght equal peak strength.
            }else{//init_stress_flag[p]==1
                Tu[p] = 1.0*Tu0;
            }
        }
//to change weakening curve while simulating, need to have a copy of Tu[p], name as Tu_0, 2017.03.11
        Tu_0[p] = Tu[p];
        for(k=0;k<=MAXTIME;k++){
            rate[k][p]=0.0;
        }
        slip_all[0][p] = 0.0/UnitofD;
        flag[p] = 0;
        flag_1stepbefore[p] = 0;
    }
    
    if(my_rank == 0){
        sprintf(path,"%s/Tu_init.txt",folder_path);
        FILE *fp = fopen(path,"w");
        for(p=0;p<ElementSize;p++){
            fprintf(fp,"%.15e\n",Tu[p]);
        }
        fclose(fp);
        
        sprintf(path,"%s/Stress0_init.txt",folder_path);
        fp = fopen(path,"w");
        for(p=0;p<ElementSize;p++){
            fprintf(fp,"%.15e\n",stress[p]);
        }
        fclose(fp);
    }
    
    
    //computing rupture processes
    //Allocate variables that will be used
    double Kerneltemp = 0.0, temp = 0.0;
    char temp_write[56];
    double sum1=0, sum2=0, pq_distance=0, theo_s_arrival_step=0, timestep_threhold=0;
    int inner_loop_time=0, approximate_flag=0, StaticRange_flag=0;
    
    
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
        local_ElementSize = local_top-local_bott+1;
    }
    
    //local array initiation
    int core_num, count_gathering=0, temp_recv, count_recv, flag_recv=0;
    int slave_size[comm_sz];
    double temp_record=0.0;
    double *local_ratek = (double*)malloc(sizeof(double)*local_ElementSize);
    double local_ratekp;
    double *local_slip = (double*)malloc(sizeof(double)*local_ElementSize);
    double local_slipp;
    double *local_stress = (double*)malloc(sizeof(double)*local_ElementSize);
    double local_stressp;
    int *local_flag = (int*)malloc(sizeof(int)*local_ElementSize);
    int local_flagp = 0;
    int local_renew_flag=0;
    
    //slave cores told the master the local_ElementSize.
    if(my_rank != 0){
        MPI_Send(&local_ElementSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else{ //my_rank == 0
        slave_size[0] = local_ElementSize;
        for(core_num=1;core_num<comm_sz;core_num++){
            MPI_Recv(&temp_recv, 1, MPI_INT, core_num, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            slave_size[core_num] = temp_recv;
        }
    }
    if(my_rank == 0){
        for(i=0;i<comm_sz;i++)
        printf("slave_size[%d] = %d\n",i,slave_size[i]);
    }
    
    //loading Kernel. different from "not using symmetricity" version, here we read in all.
    int local_Kernel_size = ElementSize*(MAXTIME+1);
    double *local_Kernel = (double*)malloc(sizeof(double)*local_Kernel_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_File fh;
    sprintf(path,"%s/kernel.dat",folder_path);
    int FILE_FLAG = MPI_File_open(MPI_COMM_WORLD,path,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
    if(fh) printf("open ok\n");
    
    //read in all, no offset.
    MPI_File_read_at(fh,0,local_Kernel,local_Kernel_size,MPI_DOUBLE,MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
    
    //computing rupture
    int kernel_index;
    printf("now core %d start to compute\n", my_rank);
    for(k=1;k<=MAXTIME;k++){
        MPI_Barrier(MPI_COMM_WORLD);
        
        //record the current_flag;
        for(p=0;p<ElementSize;p++){
            flag_1stepbefore[p] = flag[p];
        }
        
        if(my_rank == 0){
            printf("now core %d computing k=%d\n",my_rank,k);//timing
        }
        local_count = -1;
        local_start_mpi = MPI_Wtime();
        for(p=local_bott;p<=local_top;p++){
            
            sum1 = 0;
            sum2 = 0;
            local_flagp = flag[p];
            i=E[p].len;
            j=E[p].wid;
            
            for(q=0;q<ElementSize;q++){
                l=E[q].len;
                m=E[q].wid;
                /******Using the sparseness of kernel to speed up the computation of stress****/
                /******2017.03.15****/
                if(flag_1stepbefore[q]==0) continue;
                pq_distance = sqrt((x1[i]-x1[l])*(x1[i]-x1[l])+(x2[j]-x2[m])*(x2[j]-x2[m]));
                theo_s_arrival_step = (pq_distance/ct)/DELTA_T;
                timestep_threhold = max_s_influence_timestep + (int)(ceil(theo_s_arrival_step));
                if(k<=timestep_threhold){
                    approximate_flag = 0;
                }else{//k>timestep_threhold
                    approximate_flag = 1;
                }
                if(pq_distance>StaticStressRadius*DELTA_S){
                    StaticRange_flag = 0;
                }else{//pq_distance<=StaticStressRadius
                    StaticRange_flag = 1;
                }
                //calculate dynamic effect
                for(n=k-1;n>=0;n--){
                    if((k-n)>timestep_threhold){
                        //printf("core %d reach here!!!\n", my_rank);
                        break;
                    }//no dynamic effect any more;
                    kernel_index = (abs(i-l)*(WID+1)+abs(j-m))*(MAXTIME+1)+(k-n);
                    Kerneltemp = local_Kernel[kernel_index];
                    if(Kerneltemp!=0){
                        sum1 = sum1+Kerneltemp*rate[n][q];
                    }
                }
                ////calculate static effect
                if(approximate_flag == 1){
                    if(StaticRange_flag == 1){
                        //move one time step forward
                        int time_index = timestep_threhold+1;
                        kernel_index = (abs(i-l)*(WID+1)+abs(j-m))*(MAXTIME+1)+time_index;
                        Kerneltemp = local_Kernel[kernel_index];
                        
                        if(k-time_index<0) printf("MEMORY ERROR!!! k-time_index=%d, my_rank=%d\n",k-time_index,my_rank);
                        
                        sum1 += Kerneltemp*slip_all[k-time_index][q]/DELTA_T;
                    }//else StaticRange_flag == 0, do nothing
                }
                /******Using the sparseness of kernel to speed up the computation of stress****/
            }
            for(n=0;n<=(k-1);n++){
                sum2 = sum2+rate[n][p]*DELTA_T; //slip at k-1,i,j
            }
            /******setting property********/
            //modified on 2017.03.11
            if(init_stress_flag[p]==1){
                eta = eta_patch;
                Tres = Tres_0;//patches, strenght equal residue strength.
            }else{//init_stress_flag==0
                eta = eta_back;
                //Tres = Tu_0[p];
                Tres = Tres_0;
            }
//            if(local_flagp == 1){
//                Tu[p] = Tres+eta*rate[k-1][p];
//            }else{//local_flagp == 0
//                Tu[p] = Tu_0[p]+eta*rate[k-1][p];
//            }
            /******setting property********/
            temp = stress0[p]+sum1-Tu[p];
            if(temp<=0 && flag[p]==0){
                local_ratekp = 0.0;
                local_slipp = 0.0;
                local_stressp = stress0[p]+sum1;
            }
            else{
                local_flagp = 1;
//                local_ratekp = (stress0[p]+sum1-(Tu[p]-Tres)*(1-sum2/D0)+Tres)/(mu/(2.0*ct)-Tu[p]/D0*DELTA_T);//adding -eta 2017.02.22 // remove it 2017.03.11
//                if(local_ratekp<=0){
////                    printf("local_ratekp == %lf\n",local_ratekp);
////                    printf("flag_1stepbefore[p] == %d\n",flag_1stepbefore[p]);
//                    local_ratekp = 0;
//                    local_stressp = stress0[p]+sum1;
//                }
//                else{
//                    local_slipp = sum2+local_ratekp*DELTA_T;
//                    local_stressp = (Tu[p]-Tres)*(1-slip[p]/D0)+Tres;//adding +eta*local_ratekp 2017.02.22; remove it 2017.03.11
//                }
//                if(local_slipp>=D0){
                if(0==0){
                    //local_stressp = Tres;
                    local_ratekp = (1/(mu/(2.0*ct)+eta))*(stress0[p]+sum1-Tres);
                    // equation 71 in Tada 2009, page 239; when slip has exceeded D0, traction-free boundary condition is used. //changing  2.0*ct/mu to 1/(mu/(2.0*ct)-eta) 2017.02.22 //change it back 2017.03.11
                    local_stressp = Tres+eta*local_ratekp;//2017.03.11
                    if((local_ratekp<=0)){
                        //printf("local_ratekp = %lf\n",local_ratekp);
                        local_ratekp = 0;
                        local_stressp = stress0[p]+sum1;
                        local_slipp = sum2;
                    }
                    else{
                        //local_slipp = sum2+rate[k][p]*DELTA_T;
                        if(local_ratekp*DELTA_T==0){
                            printf("local_ratekp*DELTA_T = %e\n",local_ratekp*DELTA_T);
                        }
                        local_slipp = sum2+local_ratekp*DELTA_T;
                    }
                }
                if(local_slipp<slip_all[k-1][p]){
                    //("slip_all[k-1][p] = %lf, local_slipp = %lf\n",slip_all[k-1][p],local_slipp);
                }
//                if(k == 1 && p == 0 && temp>0){
//                    printf("temp = %lf + %lf - %lf\n", stress0[p], sum1, Tu[p]);
//                    return 0;
//                }
            }
//            finish = clock();
//            duration = (double)(finish-start)/CLOCKS_PER_SEC;
//            printf("k=%d,p=%d step using %lf sec\n",k,p,duration);
            local_count++;//first num should be 0. if wrong, check the initiation;
            local_ratek[local_count] = local_ratekp;
            local_slip[local_count] = local_slipp;
            local_stress[local_count] = local_stressp;
            local_flag[local_count] = local_flagp;
            
        }
 //       printf("\n\ncore %d IS COMING!!!\n\n", my_rank);
        //send local to 0 for processing
        if(my_rank != 0){
 //           printf("now core %d start to send message\n", my_rank);
            MPI_Send(local_ratek, local_ElementSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(local_slip, local_ElementSize, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            MPI_Send(local_stress, local_ElementSize, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            MPI_Send(local_flag, local_ElementSize, MPI_INT, 0, 4, MPI_COMM_WORLD);
 //           printf("core %d send successfully\n", my_rank);
        }
        //0 gather and process the data
        else{ //my_rank == 0
            count_gathering=-1;
            for(i=0;i<slave_size[0];i++){
                count_gathering++;
                ratek[count_gathering] = local_ratek[i];
                slip[count_gathering] = local_slip[i];
                stress[count_gathering] = local_stress[i];
                flag[count_gathering] = local_flag[i];
            }
            printf("core 0 writing itself ok\n");
            for(core_num=1; core_num<comm_sz; core_num++){
                free(local_ratek);
                free(local_slip);
                free(local_stress);
                free(local_flag);
                local_ratek = (double*)malloc(sizeof(double)*slave_size[core_num]);
                local_slip = (double*)malloc(sizeof(double)*slave_size[core_num]);
                local_stress = (double*)malloc(sizeof(double)*slave_size[core_num]);
                local_flag = (int*)malloc(sizeof(int)*slave_size[core_num]);
                
                MPI_Recv(local_ratek, slave_size[core_num], MPI_DOUBLE, core_num, 1, MPI_COMM_WORLD, &status);
//                MPI_Get_count(&status, MPI_DOUBLE, &count_recv);
//                if(count_recv != slave_size[core_num] && flag_recv == 0){
//                    flag_recv = 1;
//                }
                
                MPI_Recv(local_slip, slave_size[core_num], MPI_DOUBLE, core_num, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_stress, slave_size[core_num], MPI_DOUBLE, core_num, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(local_flag, slave_size[core_num], MPI_INT, core_num, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("core 0 receive from core %d successfully\n", core_num);
                for(i=0;i<slave_size[core_num];i++){
                    count_gathering++;
                    ratek[count_gathering] = local_ratek[i];
                    slip[count_gathering] = local_slip[i];
                    stress[count_gathering] = local_stress[i];
                    flag[count_gathering] = local_flag[i];
                }
            }
            if(count_gathering+1 != ElementSize){ //error, exit
                printf("core 0: I'M LEAVING!\n");
                return 0;
            }
            free(local_ratek);
            free(local_slip);
            free(local_stress);
            free(local_flag);
            local_ratek = (double*)malloc(sizeof(double)*slave_size[0]);
            local_slip = (double*)malloc(sizeof(double)*slave_size[0]);
            local_stress = (double*)malloc(sizeof(double)*slave_size[0]);
            local_flag = (int*)malloc(sizeof(int)*slave_size[0]);
        }
        //master core do the filtering
        for(p=0;p<ElementSize;p++){
            
            //only smooth in old rupture area
            if(flag_1stepbefore[p] == 0) continue;
            
            double temp_rate_point=0, temp_rate_side=0;
            
            i=E[p].len;
            j=E[p].wid;
            
            if(i==0 || i==LEN || j==0 || j==WID){//boundary, do not perform smoothing.
                ratek_filter[p] = ratek[p];
            }else{
                //point neibour rate average
                temp_rate_point = (ratek[(i-1)*(WID+1)+(j-1)] + ratek[(i+1)*(WID+1)+(j-1)] + ratek[(i-1)*(WID+1)+(j+1)] + ratek[(i+1)*(WID+1)+(j+1)])/4;
                //side neibour rate average
                temp_rate_side = (ratek[(i-1)*(WID+1)+j] + ratek[(i+1)*(WID+1)+j] + ratek[i*(WID+1)+(j+1)] + ratek[i*(WID+1)+(j-1)])/4;
                //smoothing
                ratek_filter[p] = ratek[p]*weigh_self + temp_rate_point*weigh_point + temp_rate_side*weigh_side;
            }
        }
        for(p=0;p<ElementSize;p++){
            ratek[p] = ratek_filter[p];
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(ratek, ElementSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(slip, ElementSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(stress, ElementSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(flag, ElementSize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //every core renew the rate
        for(p=0;p<ElementSize;p++){
            rate[k][p] = ratek[p];
            flag_proc[k][p] = flag[p];
            slip_all[k][p] = slip[p];
            stress_all[k][p] = stress[p];
        }
        // I am not sure whether I have to synchronize beyond every core. I guess It is no need to do that. 2016.5.12
        // [1 hour later] YES!!!!! IT MUST BE SYNCHRONIZE
        
        local_finish_mpi = MPI_Wtime();
        
        local_elapsed = (local_finish_mpi-local_start_mpi);
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if(my_rank == 0){
            printf("using %lf sec (computed by mpi.h)\n",elapsed);
        }
        
    }
    
    
    
    //writen the result. in the test stage, only rate was written
    if(my_rank == 0){
        FILE *fp_write, *fp_write2, *fp_write3, *fp_write4;
        sprintf(path,"%s/rate.dat",folder_path);
        fp_write = fopen(path,"w");
        
        sprintf(path,"%s/flag.dat",folder_path);
        fp_write2 = fopen(path,"w");
        
        sprintf(path,"%s/slip.dat",folder_path);
        fp_write3 = fopen(path,"w");
        
        sprintf(path,"%s/stress.dat",folder_path);
        fp_write4 = fopen(path,"w");
        
//        for(k=0;k<=MAXTIME;k++){
//            for(p=0;p<ElementSize;p++){
//                //write into file
//                fprintf(fp_write,"%.15e\n",rate[k][p]);
//                fprintf(fp_write2,"%d\n",flag_proc[k][p]);
//                fprintf(fp_write3,"%.15e\n",slip_all[k][p]);
//                fprintf(fp_write4,"%.15e\n",stress_all[k][p]);
//            }
//        }
        int actual_writingnum=0;
        //rate writing
        actual_writingnum = fwrite(rate,sizeof(double),ElementSize*(MAXTIME+1),fp_write);
        if(actual_writingnum!=ElementSize*(MAXTIME+1)) printf("ERROR!!!\n");
        //flag writing
        actual_writingnum = fwrite(flag_proc,sizeof(int),ElementSize*(MAXTIME+1),fp_write2);
        if(actual_writingnum!=ElementSize*(MAXTIME+1)) printf("ERROR!!!\n");
        //slip writing
        actual_writingnum = fwrite(slip_all,sizeof(double),ElementSize*(MAXTIME+1),fp_write3);
        if(actual_writingnum!=ElementSize*(MAXTIME+1)) printf("ERROR!!!\n");
        //stress writing
        actual_writingnum = fwrite(stress_all,sizeof(double),ElementSize*(MAXTIME+1),fp_write4);
        if(actual_writingnum!=ElementSize*(MAXTIME+1)) printf("ERROR!!!\n");
        
        fclose(fp_write);
        fclose(fp_write2);
        fclose(fp_write3);
        fclose(fp_write4);
    }

    
    local_finish_all = MPI_Wtime();
    local_elapsed = (local_finish_all-local_start_all);
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(my_rank == 0){
        printf("entire process using %lf sec (computed by mpi.h)\n",elapsed);
//        printf("flag_recv = %d\n",flag_recv);
    }

    MPI_Finalize();
    
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
