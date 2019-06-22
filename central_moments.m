function cm = central_moments( data ,xnorm,ynorm,p,q)
    
    cm = sum(sum((xnorm.^p).*(ynorm.^q).*data));
    cm_00 = sum(sum(data)); %this is same as mu(0,0);
    % normalise moments for scale invariance
    cm = cm/(cm_00^(1+(p+q)/2));
    
end