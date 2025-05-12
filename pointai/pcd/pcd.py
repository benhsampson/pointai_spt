from dataclasses import dataclass
from typing import Optional
import pyproj
import numpy as np

@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    crs: pyproj.CRS
    classification: Optional[np.ndarray]
    intensity: Optional[np.ndarray]

class InvalidPointCloudError(Exception):
    pass

def check_point_cloud_validity(pcd:PointCloud):
    n_xyz=len(pcd.xyz)
    n_rgb=len(pcd.rgb)
    # TODO: check that the number of points != 0
    if n_xyz!=n_rgb:
        raise InvalidPointCloudError(f'Length of coords ({n_xyz}) and colors ({n_rgb}) do not match')
    if not pcd.xyz.flags.contiguous:
        raise InvalidPointCloudError('xyz not contiguous')
    if not pcd.rgb.flags.contiguous:
        raise InvalidPointCloudError('rgb not contiguous')

def convert_crs(pcd:PointCloud, *, target_crs:pyproj.CRS, in_place:bool=False) -> PointCloud:
    if pcd.crs==target_crs:
        print('No conversion needed')
        return pcd
    transformer=pyproj.Transformer.from_crs(pcd.crs,target_crs,always_xy=True)
    # se documentation for transformer.transform() on why this is necessary
    inplace_failed=pcd.xyz.dtype==np.double or not pcd.xyz.flags.c_contiguous
    xx,yy,zz=transformer.transform(pcd.xyz[:,0],pcd.xyz[:,1],pcd.xyz[:,2],errcheck=True,inplace=in_place)
    if in_place:
        if inplace_failed:
            xyz=np.c_[xx,yy,zz]
            pcd.xyz=xyz
        pcd.crs=target_crs
    else:
        xyz=np.c_[xx,yy,zz]
        pcd=PointCloud(xyz,pcd.rgb,target_crs,pcd.classification,pcd.intensity)
    check_point_cloud_validity(pcd)
    return pcd
