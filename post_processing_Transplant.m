clc; clear all, close all
warning('off','all')
addpath D:\Ashitha
addpath D:\functions\imagescn_R2008a
addpath D:\functions\interpclosed
% 12 cases
[~,case_name_list,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI testing.xlsx', 'Sheet2', 'a1:a12' );
[~,txt,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI testing.xlsx', 'Sheet2', 'b1:b12' );
[~,data_list,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI testing.xlsx', 'Sheet2', 'c1:c12' );
% 24 cases
% [~,case_name_list,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI TRAINING.xlsx', 'Sheet2', 'a116:a139' );
% [~,txt,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI TRAINING.xlsx', 'Sheet2', 'b116:b139' );
% [~,data_list,~] = xlsread('D:\Ashitha\Transplant_study\TPM for AI TRAINING.xlsx', 'Sheet2', 'c116:c139' );

savepath1 = 'D:\Ashitha\Transplant_study\Testing_data\Input_image_4C\';
savepath2 = 'D:\Ashitha\Transplant_study\Testing_data\Ref_mask\';

savepath  = 'V:\Share\Ashitha\TPM\DL_Tx_Final\';
folder_name = 'Dense_UNet_4L';
path_mask_DL = ['D:\Ashitha\Transplant_study\Testing_data\Recon_DL\' folder_name '\'];
path_mask_post = 'D:\Ashitha\Transplant_study\Testing_data\post_processing_mask\';
% files_mask_DL = dir( path_mask_DL );
% files_image = dir( datapath );
for file_num = 3%19:length(data_list)
clear CTdataStruct
load(data_list{file_num})
case_name = strcat([case_name_list{file_num} '_' txt{file_num}]);
disp([num2str(file_num) '  ' case_name])
for slice_num = 3%1:3
disp(['slice ' num2str(slice_num)])
clear mag_image contour_mask mask_corr mask_DL  
filename1 = strcat([path_mask_DL case_name '_slice' num2str(slice_num) '_' folder_name '.mat']);
load(filename1);
mag_image = CTdataStruct.cStructVc(slice_num).magStruct.dataAy;
% Vx_image = CTdataStruct.cStructVc(slice_num).vxStruct.dataAy;
% Vy_image = CTdataStruct.cStructVc(slice_num).vyStruct.dataAy;
% Vz_image = CTdataStruct.cStructVc(slice_num).vzStruct.dataAy;
mag_image = mag_image./max(mag_image(:));
% Velocity_image = sqrt(Vx_image.^2 + Vy_image.^2 + Vz_image.^2);
% Velocity_image = Velocity_image./max(Velocity_image(:));
figure; imagescn(Vx_image(:,:,:),[],[],[],3); 
colorbar
% UNet_Dice_HD2 = UNet_4C_dice;
% filename_trans = strcat(['D:\Ashitha\Transplant_study\Testing_data\Recon_DL\UNet_Dice_HD2\' case_name '_slice' num2str(slice_num) '_UNet_Dice_HD2.mat']);
% save(filename_trans,'UNet_Dice_HD2','-v7.3');
[nx,ny,nt] = size(mag_image);

mask_DL = detect_major_obj(permute(double(Dense_UNet_4L),[2 3 1]));
if nx<ny
mask_DL = rot90(mask_DL,3);
end
if ny >=128
mask_DL = padarray(mask_DL,[abs(nx-128)/2 abs(ny-128)/2],0,'both');%abs(ny-128)/2
else
mask_DL = padarray(mask_DL,[abs(nx-128)/2 0],0,'both');%abs(ny-128)/2
mask_DL = mask_DL(:,(128-ny)/2+1:(128+ny)/2,:);
end
for t_frame =  1% 1:33 % nt
    image_temp = mag_image(:,:,t_frame);
    contour1 = CTdataStruct.sStructVc(slice_num).epiSnakeAy(t_frame).splineMx;%nodesMx  splineMx
    contour2 = CTdataStruct.sStructVc(slice_num).endoSnakeAy(t_frame).splineMx;
    LV_epi  = roipoly(image_temp,contour1(:,1),contour1(:,2));
    LV_endo = roipoly(image_temp,contour2(:,1),contour2(:,2));
    LV_epi = LV_epi - LV_endo;
    LV_bdp = double(LV_endo);
    contour3 = CTdataStruct.sStructVc(slice_num).epiSnakeAyRV(t_frame).splineMx;
    contour4 = CTdataStruct.sStructVc(slice_num).endoSnakeAyRV(t_frame).splineMx;
    RV_epi  = roipoly(image_temp,contour3(:,1),contour3(:,2));
    RV_endo = roipoly(image_temp,contour4(:,1),contour4(:,2)); 
    RV_bdp = max(RV_endo - LV_epi,0);
    RV_myo = RV_epi - RV_endo;
    RV_myo = max(RV_myo-LV_epi,0);
    mask_temp = LV_bdp + LV_epi*3 + RV_bdp + RV_myo*2 ; 
    contour_mask(:,:,t_frame)= min(mask_temp,3);
clear LV_epi LV_endo LV_myo LV_epi_in LV_BP LV_BP_out RV_epi RV_endo RV_epi_in RV_epi_out
%% post processing
Mask_DL_temp = mask_DL(:,:,t_frame);
% figure; imagescn(Mask_DL_temp,[0 3],[],[],3); 
[LVepi_y,LVepi_x] = find (detect_major_objects_2D(Mask_DL_temp == 3,1));
j = boundary(LVepi_x,LVepi_y,0);
clear B_LVepi LV_myo
B_LVepi(:,1) = LVepi_x(j);
B_LVepi(:,2) = LVepi_y(j); 
LV_epi = polyshape(B_LVepi(:,2),B_LVepi(:,1));
%     if slice_num == 3    
LV_epi_mask = polygon_to_mask(LV_epi,nx,ny);
%% LV epi
[B_BP,L_BP] = bwboundaries(detect_major_objects_2D(Mask_DL_temp == 1,2),'noholes');
% if only one blood pool
if length(B_BP)==1 
%     B_LVBP = B_BP{1}(:,[2 1]);
%     LV_myo_mask = detect_major_objects_2D(Mask_DL_temp == 3,1);
%     Mask_DL_temp(find(Mask_DL_temp ==3)) = 2;
%     Mask_DL_temp(find(LV_myo_mask)) = 3;
     %% fix valves
    [LVepi_y,LVepi_x] = find (detect_major_objects_2D(Mask_DL_temp == 3,1));
    j = boundary(LVepi_x,LVepi_y,0);
    clear B_LVepi LV_myo
    B_LVepi(:,1) = LVepi_x(j);
    B_LVepi(:,2) = LVepi_y(j); 
    LV_epi = polyshape(B_LVepi(:,2),B_LVepi(:,1));
%     if slice_num == 3    
        LV_epi_mask = polygon_to_mask(LV_epi,nx,ny);
        LV_epi_in = polybuffer(LV_epi,-3);
        LV_epi_in_mask = polygon_to_mask(LV_epi_in,nx,ny)>0;
        LV_myo_mask = (LV_epi_mask - LV_epi_in_mask)>0;
        clear mask_temp
        mask_temp = Mask_DL_temp;
        mask_temp(find(LV_myo_mask>0)) = 3;
        [B_BP_test,L_BP_test] = bwboundaries(detect_major_objects_2D(mask_temp == 1,2),'noholes');
        clear mask_temp
    if length(B_BP_test)==2    
        Mask_DL_temp(find(LV_myo_mask>0)) = 3;
        [B_BP,L_BP] = bwboundaries(detect_major_objects_2D(Mask_DL_temp == 1,2),'noholes');
        B_LVBP = B_BP{2}(:,[2 1]);
    else  %% fix apical slices
        LV_endo = polybuffer(LV_epi,-5);
        LV_endo_mask = polygon_to_mask(LV_endo,nx,ny);
        Mask_DL_temp(find(LV_endo_mask>0)) = 1;
        [B_BP,L_BP] = bwboundaries(detect_major_objects_2D(Mask_DL_temp == 1,2),'noholes');
        if length(B_BP)==2
            B_LVBP = B_BP{2}(:,[2 1]);
        else
            B_LVBP = B_BP{1}(:,[2 1]);
        end
    end
% if both blood pools are well defined
elseif length(B_BP)==2
    B_LVBP = B_BP{2}(:,[2 1]);
    LV_myo_mask = detect_major_objects_2D(Mask_DL_temp == 3,1);
    Mask_DL_temp(find(Mask_DL_temp ==3)) = 2;
    Mask_DL_temp(find(LV_myo_mask>0)) = 3;
end
LV_BP = polyshape(B_LVBP(:,2),B_LVBP(:,1));
LV_BP_out = polybuffer(LV_BP,2.5);
LV_BP_out_mask = polygon_to_mask(LV_BP_out,nx,ny);
LV_epi_mask = (LV_BP_out_mask + LV_myo_mask) >0;
B_LVepi = bwboundaries(LV_epi_mask,'noholes');
% B_LVepi = B_LVepi{1}(:,[2 1]);
B_LVepi = pick_larger_cell(B_LVepi);
nodes_BD1 = interpclosed(B_LVepi(:,1),B_LVepi(:,2),0:0.05:1,'pchip')'; % 20 points
spline_BD1 = interpclosed(nodes_BD1(:,1),nodes_BD1(:,2),0:0.01:1,'spline')'; % 200 points for spline
%% LV endo
LV_epi = polyshape(spline_BD1(:,2),spline_BD1(:,1));
LV_epi_in = polybuffer(LV_epi,-4);
LV_epi_in_mask = polygon_to_mask(LV_epi_in,nx,ny);
LV_endo_mask = double(((Mask_DL_temp == 1) + LV_epi_in_mask) >1);
B_LVendo = bwboundaries(detect_major_objects_2D(LV_endo_mask,1));
B_LVendo = B_LVendo{1}(:,[2 1]);
nodes_BD2 = interpclosed(B_LVendo(:,1),B_LVendo(:,2),0:0.05:1,'pchip')';
spline_BD2 = interpclosed(nodes_BD2(:,1),nodes_BD2(:,2),0:0.01:1,'spline')';
% figure; imagescn(LV_endo_mask,[0 3],[],[],3);
%% RV epi &  RV endo
RV_epi_mask = detect_major_objects_2D( double(((Mask_DL_temp > 0) - LV_epi_mask) >0),1);
clear epi
[RVepi_y,RVepi_x] = find (detect_major_objects_2D(RV_epi_mask>0,1));
k = boundary(RVepi_x,RVepi_y);
epi(:,1) = RVepi_x(k);
epi(:,2) = RVepi_y(k); 
epi = polyshape(epi(:,2),epi(:,1));
epi_mask = polygon_to_mask(epi,nx,ny);
RV_epi_mask = max(RV_epi_mask,epi_mask);
% figure; imagescn(RV_epi_mask,[0 3],[],[],3); 
%%
RV_BP_mask =  ((Mask_DL_temp == 1) - LV_epi_mask) >0;
% LV_myo_mask = (LV_epi_mask - LV_endo_mask)>0;
B_RV_out = bwboundaries(RV_epi_mask,'noholes');
B_RV_out = B_RV_out{1}(:,[2 1]);
RV_epi = polyshape(B_RV_out(:,2),B_RV_out(:,1));
RV_epi_out = polybuffer(RV_epi,4);
RV_epi_out_mask = polygon_to_mask(RV_epi_out,nx,ny);

if sum(RV_BP_mask(:))> 15
    B_RVBP = bwboundaries(RV_BP_mask,'noholes');
    B_RVBP_1 = B_RVBP{1}(:,[2 1]);
    RV_BP = polyshape(B_RVBP_1(:,2),B_RVBP_1(:,1));
    RV_BP_out = polybuffer(RV_BP,3);
    RV_BP_out_mask = polygon_to_mask(RV_BP_out,nx,ny);
    if length(B_RVBP)== 2 && length(B_RVBP{2})>5
        clear B_RVBP_2
        B_RVBP_2 = B_RVBP{2}(:,[2 1]);
        RV_BP_2 = polyshape(B_RVBP_2(:,2),B_RVBP_2(:,1));
        RV_BP_out_2 = polybuffer(RV_BP_2,3);
        RV_BP_out_mask_2 = polygon_to_mask(RV_BP_out_2,nx,ny);
        RV_BP_out_mask = (RV_BP_out_mask + RV_BP_out_mask_2)>0;
    end
    RV_out_mask = (RV_epi_mask + RV_BP_out_mask + ((RV_epi_out_mask + LV_myo_mask)==2))>0;
    B_RVepi = bwboundaries(RV_out_mask,'noholes');
    B_RVepi = B_RVepi{1}(:,[2 1]);
    nodes_BD3 = interpclosed(B_RVepi(:,1),B_RVepi(:,2),0:max(1/(size(B_RVepi,1)/2.5),0.05):1,'pchip')';
    spline_BD3 = interpclosed(nodes_BD3(:,1),nodes_BD3(:,2),0:0.01:1,'spline')';
    B_RVendo = bwboundaries(detect_major_objects_2D(RV_BP_mask,1),'noholes');
    B_RVendo = B_RVendo{1}(:,[2 1]);
    nodes_BD4 = interpclosed(B_RVendo(:,1),B_RVendo(:,2),0:max(1/(size(B_RVendo,1)/2.5),0.05):1,'pchip')';
    spline_BD4 = interpclosed(nodes_BD4(:,1),nodes_BD4(:,2),0:0.01:1,'spline')';
else
    RV_out_mask = (RV_epi_mask + ((RV_epi_out_mask + LV_myo_mask)==2))>0;
    B_RVepi = bwboundaries(RV_out_mask,'noholes');
    B_RVepi = B_RVepi{1}(:,[2 1]);
    nodes_BD3 = interpclosed(B_RVepi(:,1),B_RVepi(:,2),0:max(1/(size(B_RVepi,1)/2.5),0.05):1,'pchip')';
    spline_BD3 = interpclosed(nodes_BD3(:,1),nodes_BD3(:,2),0:0.01:1,'spline')';
    RV_epi_in = polyshape(B_RVepi(:,2),B_RVepi(:,1));
    RV_endo = polybuffer(RV_epi_in,-3);
    RV_endo_mask = polygon_to_mask(RV_endo,nx,ny);
    margin = -3;
    RV_BP_mask = (RV_endo_mask - LV_epi_mask)>0;
    while sum(RV_BP_mask(:))<10 && margin<-1
%    disp('changing RV_endo')
        margin = margin + 0.25;
        RV_endo = polybuffer(RV_epi_in,margin);
        RV_endo_mask = polygon_to_mask(RV_endo,nx,ny);
        RV_BP_mask = (RV_endo_mask - LV_epi_mask)>0;
    end
    if sum(RV_BP_mask(:))>10
        B_RVendo = bwboundaries(detect_major_objects_2D(RV_BP_mask,1),'noholes');
        B_RVendo = B_RVendo{1}(:,[2 1]);
        nodes_BD4 = interpclosed(B_RVendo(:,1),B_RVendo(:,2),0:max(1/(size(B_RVendo,1)/2.5),0.05):1,'pchip')';
        spline_BD4 = interpclosed(nodes_BD4(:,1),nodes_BD4(:,2),0:0.01:1,'spline')';
    else
        nodes_BD4 = [];
        spline_BD4 = [];
        disp(['Bad RV ' num2str(t_frame)])
    end
%     if slice_num == 1
%         nodes_BD4 = [];
%         spline_BD4 = [];
%         nodes_BD3 = [];
%         spline_BD3 = [];
%     end
end
% figure; imagescn(Mask_DL_temp,[0 3],[],[],3);
mask_corr(:,:,t_frame) = LV_epi_mask*3 - LV_endo_mask*2 + ((RV_out_mask-LV_epi_mask)>0)*2 -RV_BP_mask;
%% rewrite to the data
% figure(1); imagescn(mag_image(:,:,t_frame),[0 0.35],[],[],3); 
% hold on
% plot(spline_BD1(:,1),spline_BD1(:,2),'LineWidth',3,'Color','red')
% plot(spline_BD2(:,1),spline_BD2(:,2),'LineWidth',3,'Color','blue')
% plot(spline_BD3(:,1),spline_BD3(:,2),'LineWidth',3,'Color','green')
% plot(spline_BD4(:,1),spline_BD4(:,2),'LineWidth',3,'Color','yellow')
% hold off
% gifoutputPath = ['D:\Ashitha\Transplant_study\Movies\Case4_Slice3_DL.gif'];


figure(2); imagescn(mag_image(:,:,t_frame),[0 0.35],[],[],3); 
hold on
plot(contour1(:,1),contour1(:,2),'LineWidth',3,'Color','red')
plot(contour2(:,1),contour2(:,2),'LineWidth',3,'Color','blue')
plot(contour3(:,1),contour3(:,2),'LineWidth',3,'Color','green')
plot(contour4(:,1),contour4(:,2),'LineWidth',3,'Color','yellow')
hold off
gifoutputPath = ['D:\Ashitha\Transplant_study\Movies\Case4_Slice3_manual.gif'];

image_size = ny;
% figure; imagescn(mag_image((nx-image_size)/2+1:(nx+image_size)/2,(ny-image_size)/2+1:(ny+image_size)/2,t_frame),[0 0.35],[],[],3); 
% gifoutputPath = ['D:\Ashitha\Transplant_study\Movies\Case4_Slice3.gif'];
% 
% figure; imagescn(mask_DL((nx-image_size)/2+1:(nx+image_size)/2,(ny-image_size)/2+1:(ny+image_size)/2,t_frame),[0 3],[],[],3); 
% gifoutputPath = ['D:\Ashitha\Transplant_study\Movies\Case4_Slice3_mask_DL.gif'];
% 
% figure; imagescn(contour_mask((nx-image_size)/2+1:(nx+image_size)/2,(ny-image_size)/2+1:(ny+image_size)/2,t_frame),[0 3],[],[],3); 
% gifoutputPath = ['D:\Ashitha\Transplant_study\Movies\Case4_Slice3_mask_manual.gif'];

figure; imagescn(mask_DL((nx-image_size)/2+1:(nx+image_size)/2,(ny-image_size)/2+1:(ny+image_size)/2,t_frame),[0 3],[],[],3); 
% figure; imagescn(contour_mask((nx-image_size)/2+1:(nx+image_size)/2,(ny-image_size)/2+1:(ny+image_size)/2,t_frame),[0 3],[],[],3); 

colormap('gray')
axis off
daspect(gca,[1 1 1])
frame = getframe(gca);
%Write the GIF
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);
gifdelay = 0.033;
% Write to the GIF File
if t_frame == 1
    imwrite(imind,cm,gifoutputPath,'gif', 'Loopcount',inf,'DelayTime',gifdelay);
else
    imwrite(imind,cm,gifoutputPath,'gif','WriteMode','append','DelayTime',gifdelay);
end


CTdataStruct.sStructVc(slice_num).epiSnakeAy(t_frame).nodesMx = nodes_BD1;
CTdataStruct.sStructVc(slice_num).endoSnakeAy(t_frame).nodesMx = nodes_BD2;
CTdataStruct.sStructVc(slice_num).epiSnakeAyRV(t_frame).nodesMx = nodes_BD3;
CTdataStruct.sStructVc(slice_num).endoSnakeAyRV(t_frame).nodesMx = nodes_BD4;

CTdataStruct.sStructVc(slice_num).epiSnakeAy(t_frame).splineMx = spline_BD1;
CTdataStruct.sStructVc(slice_num).endoSnakeAy(t_frame).splineMx = spline_BD2;
CTdataStruct.sStructVc(slice_num).epiSnakeAyRV(t_frame).splineMx = spline_BD3;
CTdataStruct.sStructVc(slice_num).endoSnakeAyRV(t_frame).splineMx = spline_BD4;
% if slice_num ==1
%     CTdataStruct.sStructVc(slice_num).epiSnakeAyRV(t_frame).nodesMx = [];
%     CTdataStruct.sStructVc(slice_num).endoSnakeAyRV(t_frame).nodesMx = [];
%     CTdataStruct.sStructVc(slice_num).epiSnakeAyRV(t_frame).splineMx = [];
%     CTdataStruct.sStructVc(slice_num).endoSnakeAyRV(t_frame).splineMx = [];
% end
end
close all
% if nx<ny
% contour_mask = rot90(contour_mask);
% mag_image = rot90(mag_image);
% end
% clear im_comp
% im_comp(:,:,:,1) = mag_image*6;
% im_comp(:,:,:,2) = contour_mask;
% im_comp(:,:,:,3) = mask_DL;
% im_comp(:,:,:,4) = mask_corr;
% figure; imagescn(im_comp(:,:,:,:),[0 3],[2 2],[],3); 
% filename_post = strcat([path_mask_post case_name '_slice' num2str(slice_num) '_post_processed.mat']);
% save(filename_post,'mask_corr','-v7.3');
end
% folder = strcat([savepath,case_name,'_DL']);
% mkdir(folder)
% filename = strcat([folder '\',case_name,'_DL.mat']);
% save(filename,'CTdataStruct','-v7.3');
end
% figure; imagescn(mag_image(:,:,t_frame)*4,[0 3],[],[],3); 
% figure; imagescn(Velocity_image(:,:,t_frame)*4,[0 3],[],[],3); 
% figure; imagescn(contour_mask(:,:,t_frame),[0 3],[],[],3); 
% figure; imagescn(mask_DL(:,:,t_frame),[0 3],[],[],3); 
% figure; imagescn(mask_corr(:,:,t_frame),[0 3],[],[],3); 