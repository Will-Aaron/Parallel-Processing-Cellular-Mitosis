%This function is built to operate using arrayfun to permute across every
%MT to calculate if it contributes a force component onto a specific
%centrosome. Orchestrated to check the simplest parameters first in order
%to speed up calculation

%i is the target centrosome for the force, j is the other centrosome we are
%looking at
function [f_MT_1_GPU,f_MT_2_GPU,MT_flavor_GPU] = Sub_Spindle_Pole_Dynein_GPU(Exist,f_MT_1_GPU,f_MT_2_GPU,c_attach_GPU,MT_length_GPU,X_MT_1_GPU,X_MT_2_GPU,mt_vec_1_GPU,mt_vec_2_GPU,mt_centers_1_GPU,mt_centers_2_GPU,...
                                                mt_vel_1_GPU,mt_vel_2_GPU,centers_1_GPU,centers_2_GPU,vel_centers_1_GPU,vel_centers_2_GPU,MT_flavor_GPU,ProbRand,i,MinDBind,vf,f0_dynein,v0_dynein,LengthFac,a,b,c,xclose,yclose)

%If this MT doesn't contribute any force amount to the centrosome, then
%it's contribution is zero. Therefore, it is preemptively set to zero and only by
%passing all the conditional checks will it's contribution become nonzero.
f_MT_1_GPU = 0;
f_MT_2_GPU = 0;
%Checks existence of the MT and whether it will bind if it can, and ensures
%that the specific MT isn't bound to the target centrosome i
if Exist && (c_attach_GPU ~= i) && (ProbRand<=0.3)%ProbD
    %Checks if the length of the MT is long enough to span the distance between the target centrosome i and the 
    %other centrosome j, while using the info that the MT is bound to the
    %other centrosome j in the mt_centers data arrays
    if (MT_length_GPU >= sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2) -MinDBind)%Line 20 Spindle_Pole_Dynein from original prototype
        %mindistCtoC = sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2)
        %Checks if the MT is interpolant/facing towards the target
        %centrosome.
        %dot product
        if (real(acos(complex(mt_vec_1_GPU*(centers_1_GPU-mt_centers_1_GPU) + mt_vec_2_GPU*(centers_2_GPU-mt_centers_2_GPU))))<=pi/2)
            %Final check to see if the distMTtoi parameter is small enough
            %as calculated in the Spindle_Pole_Dynein function
            
            %Code from previous prototype
            %slope=(X_MT(2,Indexj(k))-centers(2,j))./(X_MT(1,Indexj(k))-centers(1,j));
         %a=-slope;
         %b=1;
         %c=-(X_MT(2,Indexj(k))-slope.*X_MT(1,Indexj(k)));
         %xclose=(b.*(b*centers(1,i)-a*centers(2,i))-a.*c)./(a.^2+b.^2);
         %yclose=(a.*(-b*centers(1,i)+a*centers(2,i))-b.*c)./(a.^2+b.^2);
         %distMTtoi=sqrt((xclose-centers(1,i)).^2+(yclose-centers(2,i)).^2);
         
    %New Version

         %This isn't organized well for the GPU since we're required to
         %pass preinitialized variables without useful data in order to
         %make this calculation readable. However the efficiency drop is
         %minimal compared to the improvement of the code's readability
         a = -mt_vec_1_GPU/mt_vec_2_GPU;
         c = -(X_MT_2_GPU)+a*X_MT_1_GPU;
         
         xclose=(b*(b*centers_1_GPU-a*centers_2_GPU)-a*c)/(a^2+b^2);
         yclose=(a*(-b*centers_1_GPU+a*centers_2_GPU)-b*c)/(a^2+b^2);
         
         if ((xclose-centers_1_GPU)^2+(yclose-centers_2_GPU)^2)<1
             %Dynein force interaction, commented out are the intermediary
             %steps as found in the CPU version and then combinations of
             %those intermediary steps. The only non-commented portion at
             %the end is the complete combination of all intermediate steps
             %into one big compilation. This is intended to make the
             %calculation code readable, yet also remove the need to
             %allocate any extra memory for an intermediary variable in the
             %calculation, as that allocation is costly for the GPU.

             %vd=(vel(:,i)-vel(:,j)-vf);
             %vd_1 = vel_centers_1_GPU-mt_vel_1_GPU-vf;
             %vd_2 = vel_centers_2_GPU-mt_vel_2_GPU-vf;
             
             %fi_DCL_1=f0_dynein*(1-vd_1/v0_dynein);%Eq2 in paper
             %fi_DCL_2=f0_dynein*(1-vd_2/v0_dynein);%Eq2 in paper
             
             %fi_DCL_1 = f0_dynein*(1-(vel_centers_1_GPU-mt_vel_1_GPU-vf)/v0_dynein);%Eq2 in paper
             %fi_DCL_2 = f0_dynein*(1-(vel_centers_2_GPU-mt_vel_2_GPU-vf)/v0_dynein);%Eq2 in paper
             
             %f_MT_1_GPU = -fi_DCL_1*(mt_vec_1_GPU)*exp(-MT_length_GPU/(LengthFac*(sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2))));%Eq8 in paper
             %f_MT_2_GPU = -fi_DCL_2*(mt_vec_2_GPU)*exp(-MT_length_GPU/(LengthFac*(sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2))));
             
             f_MT_1_GPU = -(f0_dynein*(1-(vel_centers_1_GPU-mt_vel_1_GPU-vf)/v0_dynein))*(mt_vec_1_GPU)*exp(-MT_length_GPU/(LengthFac*(sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2))));%Eq2 and Eq8 in paper
             f_MT_2_GPU = -(f0_dynein*(1-(vel_centers_2_GPU-mt_vel_2_GPU-vf)/v0_dynein))*(mt_vec_2_GPU)*exp(-MT_length_GPU/(LengthFac*(sqrt((centers_1_GPU-mt_centers_1_GPU)^2+(centers_2_GPU-mt_centers_2_GPU)^2))));
             
             MT_flavor_GPU = MT_flavor_GPU*2;%includes the prime factor 2 into the flavor encoding scheme for montoring transfers between microtubule states.
             
         end
        end
    end
end

end