function p = params_5s(p)
    p.numScale = 5;
    p.scaleStep = 1.0255;
    p.scalePenalty = 0.962;  % penalizes the change of scale
    p.scaleLR = 0.34;
    p.responseUp = 16; % response upsampling factor (purpose is to account for stride, but they dont have to be equal)
    p.windowing = 'cosine';
    p.wInfluence = 0.168;
end
