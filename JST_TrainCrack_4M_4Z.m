clc;clear;close all;
 
image_folder = 'D:\Aldian\Proyek\data\data40000an\Data Training\';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);

input = [];

for n = 1:total_images
    full_name= fullfile(image_folder, filenames(n).name);
    our_images = imread(full_name);
    data = rgb2gray(our_images);
    data = im2double(data);
    databw = im2bw(data);
    
    x1=[];
    x2=[];
    x3=[];
    x4=[];
    x5=[];
    x6=[];
    x7=[];
    x8=[];
    fiturICZ1=[];
    fiturICZ2=[];
    fiturICZ3=[];
    fiturICZ4=[];
    fiturZCZ1=[];
    fiturZCZ2=[];
    fiturZCZ3=[];
    fiturZCZ4=[];

    jarak_zona1_ICZ=[];
    jarak_zona2_ICZ=[];
    jarak_zona3_ICZ=[];
    jarak_zona4_ICZ=[];
    jarak_zona1_ZCZ=[];
    jarak_zona2_ZCZ=[];
    jarak_zona3_ZCZ=[];
    jarak_zona4_ZCZ=[];
    
    [height, width] = size(data);
    %% Perhitungan Momen Central dan Normalisasi Momen Central
    % define a co-ordinate system for image 
    xgrid = repmat((-floor(height/2):1:ceil(height/2)-1)',1,width);
    ygrid = repmat(-floor(width/2):1:ceil(width/2)-1,height,1);

    [x_bar, y_bar] = centerOfMass(data,xgrid,ygrid);

    % normalize coordinate system by subtracting mean
    % Perhitungan Orde
    xnorm = x_bar - xgrid;
    ynorm = y_bar - ygrid;

    mu_11 = central_moments( data ,xnorm,ynorm,1,1);
    mu_20 = central_moments( data ,xnorm,ynorm,2,0);
    mu_02 = central_moments( data ,xnorm,ynorm,0,2);
    mu_21 = central_moments( data ,xnorm,ynorm,2,1);
    mu_12 = central_moments( data ,xnorm,ynorm,1,2);
    mu_03 = central_moments( data ,xnorm,ynorm,0,3);
    mu_30 = central_moments( data ,xnorm,ynorm,3,0);

    %% Menghitung fitur Hu's Invariant moments
    %central_moment = [mu_11, mu_20, mu_02, mu_21, mu_12, mu_03, mu_30];
    % fitur dari momen invarian sebagai berikut:
    I_one   = mu_20 + mu_02;
    I_two   = (mu_20 - mu_02)^2 + 4*(mu_11)^2;
    I_three = (mu_30 - 3*mu_12)^2 + (mu_03 - 3*mu_21)^2;
    I_four  = (mu_30 + mu_12)^2 + (mu_03 + mu_21)^2;
    I_five  = (mu_30 - 3*mu_12)*(mu_30 + mu_12)*((mu_30 + mu_12)^2 - 3*(mu_21 + mu_03)^2) + (3*mu_21 - mu_03)*(mu_21 + mu_03)*(3*(mu_30 + mu_12)^2 - (mu_03 + mu_21)^2);
    I_six   = (mu_20 - mu_02)*((mu_30 + mu_12)^2 - (mu_21 + mu_03)^2) + 4*mu_11*(mu_30 + mu_12)*(mu_21 + mu_03);
    I_seven = (3*mu_21 - mu_03)*(mu_30 + mu_12)*((mu_30 + mu_12)^2 - 3*(mu_21 + mu_03)^2) + (mu_30 - 3*mu_12)*(mu_21 + mu_03)*(3*(mu_30 + mu_12)^2 - (mu_03 + mu_21)^2);
    
     %% Zoning ICZ
   
    barisICZ = size(databw,1);
    kolomICZ = size(databw,2);
    centroid_xICZ=round(kolomICZ/2);
    centroid_yICZ=round(barisICZ/2);

    %Pembagian Zona Citra
    I1=databw(1:size(databw,1)/2,1:size(databw,2)/2,:);
    I2=databw(size(databw,1)/2+1:size(databw,1),1:size(databw,2)/2,:);
    I3=databw(1:size(databw,1)/2,size(databw,2)/2+1:size(databw,2),:);
    I4=databw(size(databw,1)/2+1:size(databw,1),size(databw,2)/2+1:size(databw,2),:);

    %Zona 1
    baris_zona1_ICZ = size(I1,1);
    kolom_zona1_ICZ = size(I1,2);
    
    for n=1:baris_zona1_ICZ
        for m=1:kolom_zona1_ICZ
            if I1(n,m) == 1
                d_1=sqrt(((n-centroid_xICZ)^2)+((m-centroid_yICZ)^2));
                jarak_zona1_ICZ=[jarak_zona1_ICZ;d_1];
            end
        end
    end
    jarak_zona1_ICZ;
    fitur_zona1_ICZ=mean(jarak_zona1_ICZ);
    fitur_zona1_ICZ=fitur_zona1_ICZ/norm(fitur_zona1_ICZ);
    fitur_zona1_ICZ=fitur_zona1_ICZ/norm(fitur_zona1_ICZ,1);
    fitur_zona1_ICZ=fitur_zona1_ICZ/norm(fitur_zona1_ICZ,'fro');
    fitur_zona1_ICZ=fitur_zona1_ICZ/norm(fitur_zona1_ICZ,inf);

    %Zona 2
    baris_zona2_ICZ = size(I2,1);
    kolom_zona2_ICZ = size(I2,2);
    
    for n=1:baris_zona2_ICZ
        for m=1:kolom_zona2_ICZ
            if I2(n,m) == 1
                d_2=sqrt(((n-centroid_xICZ)^2)+((m-centroid_yICZ)^2));
                jarak_zona2_ICZ=[jarak_zona2_ICZ;d_2];
            end
        end
    end
    jarak_zona2_ICZ;
    fitur_zona2_ICZ=mean(jarak_zona2_ICZ);
    fitur_zona2_ICZ=fitur_zona2_ICZ/norm(fitur_zona2_ICZ);
    fitur_zona2_ICZ=fitur_zona2_ICZ/norm(fitur_zona2_ICZ,1);
    fitur_zona2_ICZ=fitur_zona2_ICZ/norm(fitur_zona2_ICZ,'fro');
    fitur_zona2_ICZ=fitur_zona2_ICZ/norm(fitur_zona2_ICZ,inf);

    %Zona 3
    baris_zona3_ICZ = size(I3,1);
    kolom_zona3_ICZ = size(I3,2);
    
    for n=1:baris_zona3_ICZ
        for m=1:kolom_zona3_ICZ
            if I3(n,m) == 1
                d_3=sqrt(((n-centroid_xICZ)^2)+((m-centroid_yICZ)^2));
                jarak_zona3_ICZ=[jarak_zona3_ICZ;d_3];
            end
        end
    end
    jarak_zona3_ICZ;
    fitur_zona3_ICZ=mean(jarak_zona3_ICZ);
    fitur_zona3_ICZ=fitur_zona3_ICZ/norm(fitur_zona3_ICZ);
    fitur_zona3_ICZ=fitur_zona3_ICZ/norm(fitur_zona3_ICZ,1);
    fitur_zona3_ICZ=fitur_zona3_ICZ/norm(fitur_zona3_ICZ,'fro');
    fitur_zona3_ICZ=fitur_zona3_ICZ/norm(fitur_zona3_ICZ,inf);

    %Zona 4
    baris_zona4_ICZ = size(I4,1);
    kolom_zona4_ICZ = size(I4,2);
    
    for n=1:baris_zona4_ICZ
        for m=1:kolom_zona4_ICZ
            if I4(n,m) == 1
                d_4=sqrt(((n-centroid_xICZ)^2)+((m-centroid_yICZ)^2));
                jarak_zona4_ICZ=[jarak_zona4_ICZ;d_4];
            end
        end
    end
    jarak_zona4_ICZ;
    fitur_zona4_ICZ=mean(jarak_zona4_ICZ);
    fitur_zona4_ICZ=fitur_zona4_ICZ/norm(fitur_zona4_ICZ);
    fitur_zona4_ICZ=fitur_zona4_ICZ/norm(fitur_zona4_ICZ,1);
    fitur_zona4_ICZ=fitur_zona4_ICZ/norm(fitur_zona4_ICZ,'fro');
    fitur_zona4_ICZ=fitur_zona4_ICZ/norm(fitur_zona4_ICZ,inf);
    
    %% Zoning ZCZ
    
    barisZCZ = size(databw,1);
    kolomZCZ = size(databw,2);
    centroid_xZCZ=round(kolomZCZ/2);
    centroid_yZCZ=round(barisZCZ/2);

    %Pembagian Zona Citra
    I1=databw(1:size(databw,1)/2,1:size(databw,2)/2,:);
    I2=databw(size(databw,1)/2+1:size(databw,1),1:size(databw,2)/2,:);
    I3=databw(1:size(databw,1)/2,size(databw,2)/2+1:size(databw,2),:);
    I4=databw(size(databw,1)/2+1:size(databw,1),size(databw,2)/2+1:size(databw,2),:);

    %Zona 1
    baris_zona1_ZCZ = size(I1,1);
    kolom_zona1_ZCZ = size(I1,2);
    centroid_x_Z1=round(kolom_zona1_ZCZ/2);
    centroid_y_Z1=round(baris_zona1_ZCZ/2);

    
    for n=1:baris_zona1_ZCZ
        for m=1:kolom_zona1_ZCZ
            if I1(n,m) == 1
                d_1=sqrt(((n-centroid_x_Z1)^2)+((m-centroid_y_Z1)^2));
                jarak_zona1_ZCZ=[jarak_zona1_ZCZ;d_1];
            end
        end
    end
    jarak_zona1_ZCZ;
    fitur_zona1_ZCZ=mean(jarak_zona1_ZCZ);
    fitur_zona1_ZCZ=fitur_zona1_ZCZ/norm(fitur_zona1_ZCZ);
    fitur_zona1_ZCZ=fitur_zona1_ZCZ/norm(fitur_zona1_ZCZ,1);
    fitur_zona1_ZCZ=fitur_zona1_ZCZ/norm(fitur_zona1_ZCZ,'fro');
    fitur_zona1_ZCZ=fitur_zona1_ZCZ/norm(fitur_zona1_ZCZ,inf);

    %Zona 2
    baris_zona2_ZCZ = size(I2,1);
    kolom_zona2_ZCZ = size(I2,2);
    centroid_x_Z2=round(kolom_zona2_ZCZ/2);
    centroid_y_Z2=round(baris_zona2_ZCZ/2);

    
    for n=1:baris_zona2_ZCZ
        for m=1:kolom_zona2_ZCZ
            if I2(n,m) == 1
                d_2=sqrt(((n-centroid_x_Z2)^2)+((m-centroid_y_Z2)^2));
                jarak_zona2_ZCZ=[jarak_zona2_ZCZ;d_2];
            end
        end
    end
    jarak_zona2_ZCZ;
    fitur_zona2_ZCZ=mean(jarak_zona2_ZCZ);
    fitur_zona2_ZCZ=fitur_zona2_ZCZ/norm(fitur_zona2_ZCZ);
    fitur_zona2_ZCZ=fitur_zona2_ZCZ/norm(fitur_zona2_ZCZ,1);
    fitur_zona2_ZCZ=fitur_zona2_ZCZ/norm(fitur_zona2_ZCZ,'fro');
    fitur_zona2_ZCZ=fitur_zona2_ZCZ/norm(fitur_zona2_ZCZ,inf);

    %Zona 3
    baris_zona3_ZCZ = size(I3,1);
    kolom_zona3_ZCZ = size(I3,2);
    centroid_x_Z3=round(kolom_zona3_ZCZ/2);
    centroid_y_Z3=round(baris_zona3_ZCZ/2);

    
    for n=1:baris_zona3_ZCZ
        for m=1:kolom_zona3_ZCZ
            if I3(n,m) == 1
                d_3=sqrt(((n-centroid_x_Z3)^2)+((m-centroid_y_Z3)^2));
                jarak_zona3_ZCZ=[jarak_zona3_ZCZ;d_3];
            end
        end
    end
    jarak_zona3_ZCZ;
    fitur_zona3_ZCZ=mean(jarak_zona3_ZCZ);
    fitur_zona3_ZCZ=fitur_zona3_ZCZ/norm(fitur_zona3_ZCZ);
    fitur_zona3_ZCZ=fitur_zona3_ZCZ/norm(fitur_zona3_ZCZ,1);
    fitur_zona3_ZCZ=fitur_zona3_ZCZ/norm(fitur_zona3_ZCZ,'fro');
    fitur_zona3_ZCZ=fitur_zona3_ZCZ/norm(fitur_zona3_ZCZ,inf);

    %Zona 4
    baris_zona4_ZCZ = size(I4,1);
    kolom_zona4_ZCZ = size(I4,2);
    centroid_x_Z4=round(kolom_zona4_ZCZ/2);
    centroid_y_Z4=round(baris_zona4_ZCZ/2);

    
    for n=1:baris_zona4_ZCZ
        for m=1:kolom_zona4_ZCZ
            if I4(n,m) == 1
                d_4=sqrt(((n-centroid_x_Z4)^2)+((m-centroid_y_Z4)^2));
                jarak_zona4_ZCZ=[jarak_zona4_ZCZ;d_4];
            end
        end
    end
    jarak_zona4_ZCZ;
    fitur_zona4_ZCZ=mean(jarak_zona4_ZCZ);
    fitur_zona4_ZCZ=fitur_zona4_ZCZ/norm(fitur_zona4_ZCZ);
    fitur_zona4_ZCZ=fitur_zona4_ZCZ/norm(fitur_zona4_ZCZ,1);
    fitur_zona4_ZCZ=fitur_zona4_ZCZ/norm(fitur_zona4_ZCZ,'fro');
    fitur_zona4_ZCZ=fitur_zona4_ZCZ/norm(fitur_zona4_ZCZ,inf);
    
    x1=[x1;I_one];
    x12=transpose(x1);

    x2=[x2;I_two];
    x22=transpose(x2);

    x3=[x3;I_three];
    x32=transpose(x3);

    x4=[x4;I_four];
    x42=transpose(x4);

    x5=[x5;I_five];
    x52=transpose(x5);

    x6=[x6;I_six];
    x62=transpose(x6);

    x7=[x7;I_seven];
    x72=transpose(x7);
    
    fiturICZ1=[fiturICZ1;fitur_zona1_ICZ];
    fiturICZ1_2=transpose(fiturICZ1);
    
    fiturICZ2=[fiturICZ2;fitur_zona2_ICZ];
    fiturICZ2_2=transpose(fiturICZ2);
    
    fiturICZ3=[fiturICZ3;fitur_zona3_ICZ];
    fiturICZ3_2=transpose(fiturICZ3);
    
    fiturICZ4=[fiturICZ4;fitur_zona4_ICZ];
    fiturICZ4_2=transpose(fiturICZ4);
    
    fiturZCZ1=[fiturZCZ1;fitur_zona1_ZCZ];
    fiturZCZ1_2=transpose(fiturZCZ1);
    
    fiturZCZ2=[fiturZCZ2;fitur_zona2_ZCZ];
    fiturZCZ2_2=transpose(fiturZCZ2);
    
    fiturZCZ3=[fiturZCZ3;fitur_zona3_ZCZ];
    fiturZCZ3_2=transpose(fiturZCZ3);
    
    fiturZCZ4=[fiturZCZ4;fitur_zona4_ZCZ];
    fiturZCZ4_2=transpose(fiturZCZ4);
    
    %input2 = [x12; x22; x32; x42; x52; x62; x72; fiturICZ1_2; fiturICZ2_2; fiturICZ3_2; fiturICZ4_2; fiturZCZ1_2; fiturZCZ2_2; fiturZCZ3_2; fiturZCZ4_2;];
    input2 = [x12 x22 x32 x42 fiturICZ1_2 fiturICZ2_2 fiturZCZ1_2 fiturZCZ2_2];
    input = [input;input2];
    input3 = transpose(input);
    
    target = zeros(1,1080); %target
    target(:,1:540) = 1;
    target(:,541:1080) = 2;
    
    %target2 = transpose(target);
    
    % = [target2];
    
end

net = newff(input3,target,[4 6 8 2],{'logsig','logsig','logsig','logsig'},'trainlm');
net.performFcn = 'mse';
net.trainParam.goal = 0.01;
net.trainParam.show = 20;
net.trainParam.epochs = 1000;
net.trainParam.mc = 0.95;
net.trainParam.lr = 0.1;
%net = train(net,input3,target);

[net,tr,Y,E] = train(net,input3,target);
 
% Hasil setelah pelatihan
bobot_hidden = net.IW{1,1};
bobot_keluaran = net.LW{2,1};
bias_hidden = net.b{1,1};
bias_keluaran = net.b{2,1};
jumlah_iterasi = tr.num_epochs;
nilai_keluaran = Y;
nilai_error = E;
error_MSE = (1/n)*sum(nilai_error.^2);
output = round(sim(net,input3));
output2 = transpose(output)

save 'D:\Aldian\Proyek\4 Hidden Layer\Model 1\net.mat' net
save 'D:\Aldian\Proyek\4 Hidden Layer\Model 1\output.mat' output2

[m,n] = find(output==target);
akurasi = sum(m)/total_images*100