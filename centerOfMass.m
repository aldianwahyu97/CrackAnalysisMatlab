function [x_bar, y_bar] = centerOfMass(data,xgrid,ygrid)

    eps = 10^(-6); % very small constant 
    
    x_bar = sum(sum((xgrid.*data)))/(sum(data(:))+eps);
    y_bar = sum(sum((ygrid.*data)))/(sum(data(:))+eps);

end