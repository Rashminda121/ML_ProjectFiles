% code here

% Read all files into matlab using for Loop
clear all
clc
close all
for nc = 1:10
T_Acc_Data_TD_Day1 = load(sprintf('U%02d_Acc_TimeD_FDay.mat', nc));
T_Acc_Data_FD_Day1 = load(sprintf('U%02d_Acc_FreqD_FDay.mat', nc));
Temp_Acc_Data_TD_D1 = T_Acc_Data_TD_Day1.Acc_TD_Feat_Vec (1:36,1:88);
Temp_Acc_Data_FD_D1 = T_Acc_Data_FD_Day1.Acc_FD_Feat_Vec (1:36,1:43);
Acc_TD_Data_Day1{nc}=Temp_Acc_Data_TD_D1;
Acc_FD_Data_Day1{nc}=Temp_Acc_Data_FD_D1;
end


