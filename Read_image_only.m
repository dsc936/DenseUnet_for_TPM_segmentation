clc; clear all; close all
addpath D:\functions\imagescn_R2008a
savepath1 = 'D:\Ashitha\data\Brad_GrantforContours\combined_image_4C\';
% savepath2 = 'D:\Ashitha\data\contour_mask_4class\';
% mkdir(savepath1)

datapath = 'D:\Ashitha\data\Brad_GrantforContours\Original_data\'; %% exclude_base  normal_data Tx
files_image = dir( datapath );
for file_num = 1:length(files_image)-2
load([datapath files_image(file_num+2).name])
case_name = files_image(file_num+2).name;
case_name = case_name(1:length(case_name)-4);
disp(case_name)
for slice_num = 1:3
clear mag_image Vx_image Vy_image Vz_image comb_image
mag_image = CTdataStruct.cStructVc(slice_num).magStruct.dataAy;
mag_image = mag_image./max(mag_image(:));
Vx_image = CTdataStruct.cStructVc(slice_num).vxStruct.dataAy;
Vy_image = CTdataStruct.cStructVc(slice_num).vyStruct.dataAy;
Vz_image = CTdataStruct.cStructVc(slice_num).vzStruct.dataAy;
% Velocity_image = sqrt(Vx_image.^2 + Vy_image.^2 + Vz_image.^2);
[nx,ny,nt] = size(mag_image);
clear comb_image_4C
comb_image_4C(:,:,:,1) = mag_image;
comb_image_4C(:,:,:,2) = Vx_image;
comb_image_4C(:,:,:,3) = Vy_image;
comb_image_4C(:,:,:,4) = Vz_image;
if nx<ny
comb_image_4C = rot90(comb_image_4C);
end
filename1 = strcat([savepath1,case_name,'_slice',num2str(slice_num),'_image.mat']);
save(filename1,'comb_image_4C','-v7.3');
end
end