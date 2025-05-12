import pathlib
from typing import TypedDict
import laspy
import numpy as np
import pyproj

from pointai.pcd import PointCloud, check_point_cloud_validity

class PointAiLasHeader(TypedDict):
    crs: pyproj.CRS

def read_las_file(path: str|pathlib.Path, *, _override_crs:pyproj.CRS|None=None) -> PointCloud:
    las=laspy.read(path)
    if _override_crs:
        crs=_override_crs
    else:
        crs=las.header.parse_crs()
        if crs is None:
            raise ValueError('`crs` not found in LAS file')
        elif not isinstance(crs,pyproj.CRS):
            raise ValueError('`crs` must be a pyproj.CRS')
    xyz:np.ndarray=np.c_[las.x,las.y,las.z]
    rgb:np.ndarray=np.c_[las.red,las.green,las.blue]
    if rgb.max() <= np.iinfo(np.uint8).max:
        rgb=rgb.astype(np.uint8)
    elif rgb.max() <= np.iinfo(np.uint16).max:
        rgb=rgb.astype(np.uint16)
    else:
        raise ValueError('`rgb` must be a uint8 or uint16 array')
    cls=np.asarray(las.classification)
    if cls is not None:
        if len(np.unique(cls)) == 1:
            cls=None
    intensity=las.intensity
    if intensity is not None:
        intensity=np.asarray(intensity)
    pcd=PointCloud(xyz,rgb,crs,cls,intensity)
    check_point_cloud_validity(pcd)
    return pcd

def read_las_header(path: str|pathlib.Path):
    with laspy.open(path, 'r') as las:
        return PointAiLasHeader(
            crs=las.header.parse_crs(), # type: ignore
        )

def write_las_file(pcd: PointCloud, dst: str|pathlib.Path, replace:bool=False, mkparent:bool=True):
    check_point_cloud_validity(pcd)
    dst=pathlib.Path(dst)
    if dst.exists() and not replace:
        raise FileExistsError(f'{dst} already exists, set `replace=True` to overwrite')
    parentdir=dst.parent
    if mkparent:
        parentdir.mkdir(parents=True, exist_ok=True)
    elif not parentdir.exists():
        raise FileNotFoundError(f'Parent directory {parentdir} does not exist')

    header = laspy.LasHeader(point_format=2, version='1.4')
    if pcd.crs:
        crs=pcd.crs
        if pcd.crs.is_compound:
            crs=crs.sub_crs_list[0]
        header.add_crs(crs)
    las = laspy.LasData(header)
    las.x = pcd.xyz[:,0]
    las.y = pcd.xyz[:,1]
    las.z = pcd.xyz[:,2]
    las.red = pcd.rgb[:,0]
    las.green = pcd.rgb[:,1]
    las.blue = pcd.rgb[:,2]
    las.write(str(dst))
    print('wrote', pcd.xyz.shape[0], 'points')