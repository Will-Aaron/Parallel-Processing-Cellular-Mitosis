    %Main Script for testing the State_All function
%function main(n_MT_init)
    %Converts input parameters
    %n_MT_init = str2double(n_MT_init);

    %%
    %Comment scheme in which certain operations reference lines from the
    %previos prototype version to denote which operations are being done in
    %this new version. Note that both a CPU and GPU version of the
    %simulation will be run simultaneously to observe computational
    %efficiency for each time step of the simulation between versions.
    %%

    n_MT_init = 300;
    %Timing Parameters to test more cases
    iterF = 600;
    timings_State = zeros(2,iterF);
    timings_Create = zeros(2,iterF);
    timings_Delete = zeros(2,iterF);
    timings_Update = zeros(2,iterF);
    timings_Force = zeros(2,iterF);
    timings_Move = zeros(2,iterF);
    
    timings_AntiP = zeros(2,iterF);
    
    
    iter = 1;
    gather_rate = 5;
    gath = 1;
    
    %initialize randomization
    generator={'twister','simdTwister','combRecursive','multFibonacci'};
    gi=randi([1 4],1);
    rng('shuffle',generator{gi});

    %This to see how many MT's are at each step
    MT_number = zeros(2,iterF);
    
    %Main Script Parameters
    nc=2;%Line 4 main_Ref from original prototype
    dt=0.5; %Line 9 main_Ref from original prototype
    %fac=0.01; %Line 11 main_Ref from original prototype
    fac = 0.15;%From Paper, catastrophe rate scaling
    net_force=zeros(2,nc); %Line 31 main_Ref from original prototype
    MaxSumLength = 10000;
    
    cputic = tic;
    initialization(n_MT_init,iterF,nc,dt,fac);
    load('initialization.mat');
    parameters;
    init_CPU = toc(cputic);
    
    gputic = tic;
    initialization_GPU;
    init_GPU = toc(gputic);
    
    centers_old=centers;
    
    while (iter<= iterF)
        
        cputic = tic;
        %CreatenewMTs CPU Version
        if(sum(MT_length)<MaxSumLength) 
        [c_attachnew mt_centersnew MT_lengthnew MT_anglenew X_MTnew mt_vecnew MT_statenew] = createnewMTs(mt_nuc, nc, c_rad, centers);
        n_MT=mt_nuc+n_MT; %update total # of MTs
        X_MT = [X_MT X_MTnew];  c_attach = [c_attach c_attachnew];  mt_centers = [mt_centers mt_centersnew];  
        MT_length = [MT_length MT_lengthnew]; MT_angle = [MT_angle MT_anglenew];  mt_vec = [mt_vec mt_vecnew];  
        MT_state = [MT_state MT_statenew]; 
        end
        timings_Create(1,iter) = toc(cputic);
        
        gputic = tic;
        %CreatenewMTs GPU Version
        if (sum(MT_length_GPU(Exist))<MaxSumLength)
            [Exist] = createnewMTs_GPU(Exist,mt_nuc);
            %n_MT_GPU = sum(Exist); %We don't actually need to know how
            %many MT's there are until the end of the iteration for data
            %purposes.
            
        end
        timings_Create(2,iter) = toc(gputic);
        
        
        
        
        cputic = tic;
        %Main Script CPU Prep States
        k2 =fac*gv_astral*dt*MT_length(1,:);
        %k2=0.0125*MT_length(1,:)/2;
        indicesg=find(MT_state==1);
        indicess=find(MT_state==2);
        indicesSlip=find(MT_state==3);
        indicesb=find(MT_state==4);
        Lg=length(indicesg);
        Ls=length(indicess);
        Lslip=length(indicesSlip);
        Lb=length(indicesb);

        %CPU Version States
        %State_1
        [MT_state] = State_1(indicesg,Lg,MinDBind,MT_state,k2,dt,xRadius,X_MT,gv_astral,probD);

        %State_2
        [MT_state] = State_2(indicess,Ls,k1,dt,MT_state);

        %State_3
        [MT_state,MT_angle,mt_vec,X_MT] = State_3(mt_vec,mt_centers,c_rad,MT_length,indicesSlip,Lslip,MinDBind,n_MT,MT_angle,MT_state,k2,dt,xRadius,X_MT,gv_astral,probD,dtheta);

        %State_4    
        [MT_state] = State_4(indicesb,Lb,MinDBind,xRadius,X_MT,MT_state,net_force,mt_vec,f0_dynein,c_attach,gv_astral,dt);

        timings_State(1,iter) = toc(cputic);

        gputic = tic;
        %Main Script GPU Prep States
        ProbRand = rand(1,upBound,'gpuArray');


        %GPU Version States
        [MT_state_GPU,MT_angle_GPU,mt_vec_1_GPU,mt_vec_2_GPU,X_MT_1_GPU,X_MT_2_GPU] = arrayfun(@State_All_GPU,Exist,MT_state_GPU,MT_angle_GPU,MT_length_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,...
                                                                                                            X_MT_1_GPU,X_MT_2_GPU,ProbRand,fac,gv_astral,MinDBind,dt,k1,xRadius,probD,c_rad,dtheta);
        timings_State(2,iter) = toc(gputic);
        
        
        %CPU Version Forces
        cputic = tic;
        %F_i_DCL
        [DCLState, F_DCL]=Spindle_Pole_Dynein(c_rad,LengthFac,MinDBind,probD,gv_astral,mt_vec,dt,nc,c_attach,X_MT,centers,centers_old,vf,f0_dynein,v0_dynein,MT_length,MT_angle);
        
        %CtoCortex-Repulsive
        [F_repulsive_centcortex] = Cent_to_Cortex_Rep(xRadius,nc,C,repd,x,y,centers);  
         
        %CtoC-Repulsive
        [F_repulsive_centc]=Cent_to_Cent_Rep(nc,centers,C,repd); 
        
        %F_stari_cortex
        [F_stari_cortex]=Slipping_Force(nc,MT_state,c_attach,f_stall,kappa,MT_length,mt_vec);
        
        %F_i_cortex
        [F_i_cortex]=Cortical_Dynein_Force(LengthFac,xRadius,nc,MT_state,c_attach,centers,centers_old,dt,vf,f0_dynein,v0_dynein,mt_vec,MT_length);
         
        timings_Force(1,iter) = toc(cputic);
        %Quick inclusion of Antiparallel_Forces
        cputic = tic;
        %[F_antiparallel_Eg5, F_antiparallel_HSET, Eg5State, HSETState, mt_vec, X_MT]=Antiparallel_ForcesV3(LengthFac,vf,clock,CrossLink,drag,X_MT,probHSET,probEg5,MT_state,incDist,MinDBind_Int,mt_vec,nc,MT_length,c_rad,mt_centers,MT_angle,c_attach,centers,centers_old,dt,f0_Eg5,v0_Eg5,f0_HSET,v0_HSET);
        [F_antiparallel_Eg5 F_antiparallel_HSET Eg5StateFinal HSETStateFinal mt_vec X_MT]=Antiparallel_ForcesV3(LengthFac,CrossLink,X_MT,probHSET,probEg5,MT_state,incDist,MinDBind_Int,mt_vec,nc,MT_length,c_rad,mt_centers,MT_angle,c_attach,centers,centers_old,dt,f0_Eg5,v0_Eg5,f0_HSET,v0_HSET);
        timings_AntiP(1,iter) = toc(cputic);
        
        %CPU Version Motion
        
        cputic = tic;
        centers_old=centers;
        
        for i = 1:nc
            %net_force(:,i)=F_stari_cortex(:,i)+F_i_cortex(:,i)+F_repulsive_centcortex(:,i)+F_repulsive_centc(:,i)+F_antiparallel_Eg5(:,i)+F_antiparallel_HSET(:,i)+F_DCL(:,i);
            net_force(:,i) =F_stari_cortex(:,i)+F_i_cortex(:,i)+F_DCL(:,i)+F_repulsive_centcortex(:,i)+F_repulsive_centc(:,i)+F_antiparallel_Eg5(:,i)+F_antiparallel_HSET(:,i);
            centers(:,i) = centers(:,i)-(dt/drag)*net_force(:,i);
        end
        
        for i = 1:n_MT
            mt_centers(:,i) = [centers(:,c_attach(i))];  
        end
        timings_Move(1,iter) = toc(cputic);
        
        %CPU Version MT_Update
        cputic = tic;
        [MT_length] = MT_update(MT_state,MT_length,dt,gv_astral,sv_astral,bsv_astral);
        timings_Update(1,iter) = toc(cputic);
        
        
        %GPU Version Forces
        gputic = tic;
        %F_DCL_1_GPU,F_DCL_2_GPU
        [F_DCL_1_GPU,F_DCL_2_GPU,MT_flavor_GPU] = Spindle_Pole_Dynein_GPU(F_DCL_1_GPU,F_DCL_2_GPU,f_MT_1_GPU,f_MT_2_GPU,Exist,c_attach_GPU,MT_length_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,...
                                                    mt_vel_1_GPU,mt_vel_2_GPU,centers_1_GPU,centers_2_GPU,vel_centers_1_GPU,vel_centers_2_GPU,MT_flavor_GPU,ProbRand,nc,MinDBind,vf,f0_dynein,v0_dynein,LengthFac,a,b,c,xclose,yclose);
        
        %F_repulsize_centcortex_1_GPU,F_repulsize_centcortex_1_GPU
        [F_repulsive_centcortex_1_GPU,F_repulsive_centcortex_2_GPU] = Cent_to_Cortex_Rep_GPU(xRadius,C,repd,x_GPU,y_GPU,centers_1_GPU,centers_2_GPU,F_repulsive_centcortex_1_GPU,F_repulsive_centcortex_2_GPU,mindistCtoCortex_GPU,II_GPU);
          
                                                
        %CtoC-repulsive
        [F_repulsive_centc_1_GPU,F_repulsive_centc_2_GPU] = Cent_to_Cent_Rep_GPU(centers_1_GPU,centers_2_GPU,F_repulsive_centc_1_GPU,F_repulsive_centc_2_GPU,nc,C,repd);
        
        %F_stari_cortex
        [F_stari_cortex_1_GPU,F_stari_cortex_2_GPU] = Slipping_Force_GPU(F_stari_cortex_1_GPU,F_stari_cortex_2_GPU,Exist,MT_state_GPU,c_attach_GPU,MT_length_GPU,mt_vec_1_GPU,mt_vec_2_GPU,f_indvMT_1_GPU,f_indvMT_2_GPU,f_stall,kappa,nc);

        %F_i_cortex
        [F_i_cortex_1_GPU,F_i_cortex_2_GPU] = Cortical_Dynein_Force_GPU(F_i_cortex_1_GPU,F_i_cortex_2_GPU,Exist,c_attach_GPU,MT_state_GPU,MT_length_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_vel_1_GPU,mt_vel_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,f_indvMT_1_GPU,f_indvMT_2_GPU,f0_dynein,v0_dynein,LengthFac,xRadius,nc);

        
        timings_Force(2,iter) = toc(gputic);
        %Calculates the MeshGrids for the Antiparallel Force Function
        gputic = tic;
        [mt_vec_1_GPU_i,mt_vec_1_GPU_j,mt_vec_2_GPU_i,mt_vec_2_GPU_j,MT_length_GPU_i,MT_length_GPU_j,mt_centers_1_GPU_i,mt_centers_1_GPU_j,mt_centers_2_GPU_i,mt_centers_2_GPU_j,mt_vel_1_GPU_i,mt_vel_1_GPU_j,...
    mt_vel_2_GPU_i,mt_vel_2_GPU_j,c_attach_GPU_i,c_attach_GPU_j,Exist_i,Exist_j,MT_flavor_GPU_i,MT_flavor_GPU_j] = makeMeshGrid(mt_vec_1_GPU_i,mt_vec_1_GPU_j,mt_vec_2_GPU_i,mt_vec_2_GPU_j,MT_length_GPU_i,MT_length_GPU_j,mt_centers_1_GPU_i,mt_centers_1_GPU_j,mt_centers_2_GPU_i,mt_centers_2_GPU_j,mt_vel_1_GPU_i,mt_vel_1_GPU_j,...
    mt_vel_2_GPU_i,mt_vel_2_GPU_j,c_attach_GPU_i,c_attach_GPU_j,Exist_i,Exist_j,MT_flavor_GPU_i,MT_flavor_GPU_j,mt_vec_1_GPU,mt_vec_2_GPU,MT_length_GPU,mt_centers_1_GPU,mt_centers_2_GPU,mt_vel_1_GPU,mt_vel_2_GPU,c_attach_GPU,Exist,MT_flavor_GPU);
        
        %Calculates the Antiparallel Forces
        [F_anti_Eg5_1_GPU,F_anti_Eg5_2_GPU,F_anti_HSET_1_GPU,F_anti_HSET_2_GPU] = Antiparallel_ForcesV3_GPU(mt_vec_1_GPU_i,mt_vec_1_GPU_j,mt_vec_2_GPU_i,mt_vec_2_GPU_j,MT_length_GPU_i,MT_length_GPU_j,mt_centers_1_GPU_i,mt_centers_1_GPU_j,mt_centers_2_GPU_i,mt_centers_2_GPU_j,mt_vel_1_GPU_i,mt_vel_1_GPU_j,...
    mt_vel_2_GPU_i,mt_vel_2_GPU_j,c_attach_GPU_i,c_attach_GPU_j,Exist_i,Exist_j,MT_flavor_GPU_i,MT_flavor_GPU_j,F_MT_1_Eg5,F_MT_2_Eg5,F_MT_1_HSET,F_MT_2_HSET,F_anti_Eg5_1_GPU,F_anti_Eg5_2_GPU,F_anti_HSET_1_GPU,F_anti_HSET_2_GPU,...
    nc,c_attach_GPU,c_rad,BindEg5,BindHSET,probEg5,probHSET,f0_Eg5,f0_HSET,v0_Eg5,v0_HSET,CrossLink,LengthFac,intAngle,t,u,RcrossS,RdiffCross,SdiffCross,overlap,v,upBound);

        timings_AntiP(2,iter) = toc(gputic);
        
%Look in Force_All_V3 folder for same model but with commands to pull and
%look at force contributions of the anti-parallel force function to varify
%validity of calculation

        %GPU Version Motion, it involves a lot of indexing so it will be
        %very slow on the GPU. Therefore design choices are made to
        %mitigate indexing wherever possible.
        gputic = tic;
        centers_1_old_GPU=centers_1_GPU;
        centers_2_old_GPU=centers_2_GPU;
        
     
        
            %net_force(:,i)=F_stari_cortex(:,i)+F_i_cortex(:,i)+F_repulsive_centcortex(:,i)+F_repulsive_centc(:,i)+F_antiparallel_Eg5(:,i)+F_antiparallel_HSET(:,i)+F_DCL(:,i);
            net_force_1_GPU = gather(F_DCL_1_GPU + F_stari_cortex_1_GPU + F_i_cortex_1_GPU + F_anti_Eg5_1_GPU + F_anti_HSET_1_GPU) + F_repulsive_centcortex_1_GPU + F_repulsive_centc_1_GPU;
            net_force_2_GPU = gather(F_DCL_2_GPU + F_stari_cortex_2_GPU + F_i_cortex_2_GPU + F_anti_Eg5_2_GPU + F_anti_HSET_2_GPU) + F_repulsive_centcortex_2_GPU + F_repulsive_centc_2_GPU;
            centers_1_GPU = centers_1_GPU-(dt/drag)*net_force_1_GPU;
            centers_2_GPU = centers_2_GPU-(dt/drag)*net_force_2_GPU;
        
        
        vel_centers_1_GPU = (centers_1_GPU - centers_1_old_GPU)/dt;
        vel_centers_2_GPU = (centers_2_GPU - centers_2_old_GPU)/dt;
        %Updates the velocities and centers of the centrosomes that each MT is attached
        %to, as well as the location data for each MT.
        for kk = 1:nc
            mt_vel_1_GPU(c_attach_GPU == kk) = vel_centers_1_GPU(kk);
            mt_vel_2_GPU(c_attach_GPU == kk) = vel_centers_2_GPU(kk);
            
            mt_centers_1_GPU(c_attach_GPU == kk) = centers_1_GPU(kk);
            mt_centers_2_GPU(c_attach_GPU == kk) = centers_2_GPU(kk);
        end
        timings_Move(2,iter) = toc(gputic);
        
        %GPU Version MT_Update
        gputic = tic;
        [MT_length_GPU] = arrayfun(@MT_update_GPU,Exist,MT_state_GPU,MT_length_GPU,dt,gv_astral,sv_astral,bsv_astral);
        timings_Update(2,iter) = toc(gputic);
        
        
        %CPU Version Delete
        cputic = tic;
        [c_attach, mt_centers, MT_length, MT_angle, X_MT, mt_vec, MT_state, n_MT] = delete_MTs(n_MT, MT_length, c_attach, mt_centers, MT_angle, X_MT, mt_vec, MT_state, c_rad);
        timings_Delete(1,iter) = toc(cputic);
        
        %GPU Verison Delete
        gputic = tic;
        [Exist,MT_length_GPU,MT_state_GPU,MT_angle_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU] = ...
            arrayfun(@delete_MTs_GPU,Exist,MT_length_GPU,MT_state_GPU,MT_angle_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,c_rad);
        n_MT_GPU = sum(Exist);
        timings_Delete(2,iter) = toc(gputic);
        
        MT_number(1,iter) = n_MT;
        MT_number(2,iter) = gather(n_MT_GPU);
        iter = iter +1;
        
        
    end

    CPU_Time_State = mean(timings_State(1,:));
    GPU_Time_State = mean(timings_State(2,:));
    timediff_State = GPU_Time_State - CPU_Time_State;
    CPU_GPU_ratio_State = CPU_Time_State / GPU_Time_State;
    
    CPU_Time_Update = mean(timings_Update(1,:));
    GPU_Time_Update = mean(timings_Update(2,:));
    timediff_Update = GPU_Time_Update - CPU_Time_Update;
    CPU_GPU_ratio_Update = CPU_Time_Update / GPU_Time_Update;
    
    CPU_Time_Delete = mean(timings_Delete(1,:));
    GPU_Time_Delete = mean(timings_Delete(2,:));
    timediff_Delete = GPU_Time_Delete - CPU_Time_Delete;
    CPU_GPU_ratio_Delete = CPU_Time_Delete / GPU_Time_Delete;
    
    CPU_Time_Create = mean(timings_Create(1,:));
    GPU_Time_Create = mean(timings_Create(2,:));
    timediff_Create = GPU_Time_Create - CPU_Time_Create;
    CPU_GPU_ratio_Create = CPU_Time_Create / GPU_Time_Create;
    
    CPU_Time_Force = mean(timings_Force(1,:));
    GPU_Time_Force = mean(timings_Force(2,:));
    timediff_Force = GPU_Time_Force - CPU_Time_Force;
    CPU_GPU_ratio_Force = CPU_Time_Force / GPU_Time_Force;
    
    CPU_Time_Move = mean(timings_Move(1,:));
    GPU_Time_Move = mean(timings_Move(2,:));
    timediff_Move = GPU_Time_Move - CPU_Time_Move;
    CPU_GPU_ratio_Move = CPU_Time_Move / GPU_Time_Move;
    
    CPU_Time_AntiP = mean(timings_AntiP(1,:));
    GPU_Time_AntiP = mean(timings_AntiP(2,:));
    timediff_AntiP = GPU_Time_AntiP - CPU_Time_AntiP;
    CPU_GPU_ratio_AntiP = CPU_Time_AntiP / GPU_Time_AntiP;
    

    timediff_iter = timediff_Create + timediff_Delete + timediff_State + timediff_Update + timediff_Force + timediff_Move + timediff_AntiP;
    
    
    %Gathers the GPU Data
    gputic = tic;
    
    %Do we really need to have these centers as GPU arrays?, Well they
    %applied are on the GPU since their output is all calculated on the
    %GPU. Theoretically I could gather the net force every time step which
    %wouldn't take longer than normal if I wanted to have results of every
    %time step, however this information, while more costly to retrieve
    %compared to the CPU version, is inconsequential when compared to the
    %speed-up of the Spindle_Pole_Dynein force calculation on GPU
    
    c_attach_GPU = gather(c_attach_GPU);
    MT_length_GPU = gather(MT_length_GPU);
    MT_angle_GPU = gather(MT_angle_GPU);
    MT_state_GPU = gather(MT_state_GPU);
    
    mt_centers_1_GPU = gather(mt_centers_1_GPU);
    mt_centers_2_GPU = gather(mt_centers_2_GPU);
    
    X_MT_1_GPU = gather(X_MT_1_GPU);
    X_MT_2_GPU = gather(X_MT_2_GPU);
    
    mt_vec_1_GPU = gather(mt_vec_1_GPU);
    mt_vec_2_GPU = gather(mt_vec_2_GPU);
    Exist = gather(Exist);
    n_MT_GPU = gather(n_MT_GPU);
    totalgathertime = toc(gputic);
    timediff_Init = init_GPU - init_CPU;
    
    total_CPU_Time = init_CPU + (CPU_Time_Create+CPU_Time_Delete+CPU_Time_Force+CPU_Time_Move+CPU_Time_State+CPU_Time_Update+CPU_Time_AntiP)*iterF;
    
    total_GPU_Time = init_GPU + (GPU_Time_Create+GPU_Time_Delete+GPU_Time_Force+GPU_Time_Move+GPU_Time_State+GPU_Time_Update+GPU_Time_AntiP)*iterF + totalgathertime*(iterF/gather_rate);
    
    total_time_diff = total_GPU_Time - total_CPU_Time;
    
    
    save('results.mat');
    save('timings.mat','timings_State','CPU_Time_State','GPU_Time_State','CPU_GPU_ratio_State','timediff_State','timings_Update','CPU_Time_Update','GPU_Time_Update','CPU_GPU_ratio_Update','timediff_Update',...
        'timings_Force','CPU_Time_Force','GPU_Time_Force','CPU_GPU_ratio_Force','timediff_Force','timings_Move','CPU_Time_Move','GPU_Time_Move','CPU_GPU_ratio_Move','timediff_Move',...
        'timings_Delete','CPU_Time_Delete','GPU_Time_Delete','CPU_GPU_ratio_Delete','timediff_Delete','timings_Create','CPU_Time_Create','GPU_Time_Create','CPU_GPU_ratio_Create','timediff_Create','timediff_iter','totalgathertime',...
        'init_CPU','init_GPU','timediff_Init','total_time_diff','total_CPU_Time','total_GPU_Time','CPU_Time_AntiP','GPU_Time_AntiP','timediff_AntiP','CPU_GPU_ratio_AntiP');
    %end