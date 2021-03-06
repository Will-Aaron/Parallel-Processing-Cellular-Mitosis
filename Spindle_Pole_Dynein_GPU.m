%This function calcualtes the force component generated by dynein acting on
%the spindle pole between respective centrosomes.

%The function uses the velocity of the centers in accordance with their
%locaion in a pervious timestep, this will be caculated earlier outside
%this force function
function [F_DCL_1_GPU,F_DCL_2_GPU,MT_flavor_GPU] = Spindle_Pole_Dynein_GPU(F_DCL_1_GPU,F_DCL_2_GPU,f_MT_1_GPU,f_MT_2_GPU,Exist,c_attach_GPU,MT_length_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,...
    mt_vel_1_GPU,mt_vel_2_GPU,centers_1_GPU,centers_2_GPU,vel_centers_1_GPU,vel_centers_2_GPU,MT_flavor_GPU,ProbRand,nc,MinDBind,vf,f0_dynein,v0_dynein,LengthFac,a,b,c,xclose,yclose)


    %Iterates over each centrosome, and sees if the microtubual will apply
    %a force to that centrosome
    %Process uses the arrayfun functionality in MATLAB for it's efficiency
    %in distributing workload to the GPU
    for i = 1:nc
        [f_MT_1_GPU(i,:),f_MT_2_GPU(i,:),MT_flavor_GPU] = arrayfun(@Sub_Spindle_Pole_Dynein_GPU,Exist,f_MT_1_GPU(i,:),f_MT_2_GPU(i,:),c_attach_GPU,MT_length_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,...
                                            mt_vel_1_GPU,mt_vel_2_GPU,centers_1_GPU(i),centers_2_GPU(i),vel_centers_1_GPU(i),vel_centers_2_GPU(i),MT_flavor_GPU,ProbRand,i,MinDBind,vf,f0_dynein,v0_dynein,LengthFac,a,b,c,xclose,yclose); 
    end
    
    %Now that the matrix containing force contributions from each MT has
    %been filled out, we iterate over that matrix, summing up the
    %contributions onto each centrosome
    for j = 1:nc
        F_DCL_1_GPU(j) = sum(f_MT_1_GPU(j,:)) + sum(sum(f_MT_1_GPU(:,c_attach_GPU==j)));
        F_DCL_2_GPU(j) = sum(f_MT_2_GPU(j,:)) + sum(sum(f_MT_2_GPU(:,c_attach_GPU==j)));
    end
    
    
end