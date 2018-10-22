kernel void mean2std(global float* value,
                     global float2* mean,
                     global float* delta2,
                     int size)
{
    int gid = get_global_id(0);
    if (gid>=size)
    {
       return;
    }
    float m = mean[0].s0/mean[0].s1;
    float delta = value[gid] - m;
    delta2[gid] = delta * delta;
}

kernel void sigmaclip(global float* value,
                       global float2* mean,
                       global float2* std,
                       float cutof,                 
                       int size)
{
    int gid = get_global_id(0);
    if (gid>=size)
    {
       return; 
    }
       
    float m = mean[0].s0 / mean[0].s1;
    float s = sqrt(std[0].s0 / (std[0].s1 - 1.0f));
    if (fabs(value[gid]-m)>(cutof*s))
    {
        value[gid] = NAN;
    }
}
