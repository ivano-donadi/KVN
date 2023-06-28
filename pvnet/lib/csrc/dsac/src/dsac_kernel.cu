#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// #include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__
void generate_hypothesis_kernel(
    float* direct,     // [tn,vn,2]
    float* coords,     // [tn,2]
    int* idxs,         // [hn,vn,2]
    float* hypo_pts,   // [hn,vn,2]
    int tn,
    int vn,
    int hn
)
{
    int hvi = threadIdx.x + blockIdx.x*blockDim.x;
    if(hvi>=hn*vn) return;

    int hi=hvi/vn;
    int vi=hvi-hi*vn;

    int t0=idxs[hi*vn*2+vi*2];
    int t1=idxs[hi*vn*2+vi*2+1];

    float nx0=direct[t0*vn*2+vi*2+1];
    float ny0=-direct[t0*vn*2+vi*2];
    float cx0=coords[t0*2];
    float cy0=coords[t0*2+1];

    float nx1=direct[t1*vn*2+vi*2+1];
    float ny1=-direct[t1*vn*2+vi*2];
    float cx1=coords[t1*2];
    float cy1=coords[t1*2+1];

    if(fabs(nx1*ny0-nx0*ny1)<1e-6) return;
    if(fabs(ny1*nx0-ny0*nx1)<1e-6) return;

    // compute intersection
    float y=(nx1*(nx0*cx0+ny0*cy0)-nx0*(nx1*cx1+ny1*cy1))/(nx1*ny0-nx0*ny1);
    float x=(ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))/(ny1*nx0-ny0*nx1);

    hypo_pts[hi*vn*2+vi*2]=x;
    hypo_pts[hi*vn*2+vi*2+1]=y;
}

at::Tensor generate_hypothesis_launcher(
    at::Tensor direct,     // [tn,vn,2]
    at::Tensor coords,     // [tn,2]
    at::Tensor idxs        // [hn,vn,2]
)
{
    int tn=direct.size(0);
    int vn=direct.size(1);
    int hn=idxs.size(0);

    assert(direct.size(2)==2);
    assert(coords.size(0)==tn);
    assert(coords.size(1)==2);
    assert(idxs.size(1)==vn);
    assert(idxs.size(2)==2);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(hn*vn,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    auto hypo_pts = at::zeros({hn,vn,2}, direct.options());
    generate_hypothesis_kernel<<<bdim,tdim>>>(
        direct.data_ptr<float>(),
        coords.data_ptr<float>(),
        idxs.data_ptr<int>(),
        hypo_pts.data_ptr<float>(),
        tn,vn,hn
    );
    gpuErrchk(cudaGetLastError())

    return hypo_pts;
}


/*
def intersection_from_dir(p0,p1,n0,n1):
    #print(p0.shape, p1.shape, n0.shape, n1.shape)
    #quit()
    cx0 = p0[:,:,0] #[rn,kn]
    cy0 = p0[:,:,1] #[rn,kn]
    cx1 = p1[:,:,0] #[rn,kn]
    cy1 = p1[:,:,1] #[rn,kn]
    nx0 = n0[:,:,1] #[rn,kn]
    ny0 = -n0[:,:,0] #[rn,kn]
    nx1 = n1[:,:,1] #[rn,kn]
    ny1 = -n1[:,:,0] #[rn,kn]
    numy = (nx1*(nx0*cx0+ny0*cy0)-nx0*(nx1*cx1+ny1*cy1))
    numx = (ny1*(nx0*cx0+ny0*cy0)-ny0*(nx1*cx1+ny1*cy1))
    deny = (nx1*ny0-nx0*ny1)
    denx = (ny1*nx0-ny0*nx1)
    y=numy/(deny)
    x=numx/(denx)

    #print(denx.max(), deny.max())

    invalidx = x.isnan().logical_or(x.isinf()).logical_or(x <-224).logical_or(x > 2*224)
    invalidy = y.isnan().logical_or(y.isinf()).logical_or(y <-126).logical_or(y > 2*126)
    invalidden = (denx.abs()<thresh).logical_or(deny.abs()<thresh)
    #print(invalidden)
    valid = (invalidx.logical_not()).logical_and((invalidy.logical_not())).logical_and(invalidden.logical_not())

    x[valid.logical_not()] = 0
    y[valid.logical_not()] = 0

    #y = suppress_nan_inf_thresh(y,deny,thresh)
    #x = suppress_nan_inf_thresh(x,denx,thresh)

    #x = limit_to_range(x,x,-224,224*2)
    #y = limit_to_range(y,y,-126,126*2)

    return x,y,valid
*/

/*
def generate_hypotheses(coords, direct, vn, num=None, idxs=None):
    assert ((num is not None) and num > 0) or (idxs is not None), "either 'num' or 'idxs' must be given"
    if idxs is None:
        idxs = torch.zeros([num,vn, 2], dtype=torch.long, device=direct.device).random_(0, direct.shape[0])
    else:
        num = idxs.shape[0]
    i0 = idxs[:,:,0]
    i1 = idxs[:,:,1]
    p0 = coords[idxs[:,:,0]] #[rn,vn,2]
    n0 = torch.cat([direct[i0[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,vn,2]
    p1 = coords[idxs[:,:,1]] #[rn,2]
    n1 = torch.cat([direct[i1[:,x],x,:].unsqueeze(1) for x in range(vn)],dim=1) #[rn,vn,2]
    x,y, valid = intersection_from_dir(p0,p1,n0,n1) #([rn,vn],[rn,vn],[rn,vn])
    cur_hyp_pts = torch.cat([x.unsqueeze(2),y.unsqueeze(2)],dim=2)
    return cur_hyp_pts, valid
*/