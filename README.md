# Task1
1. Use `python task1_load.py` to collect BEV and front view images.
  The saved images can be found under `data/task1`
2. Ues `python bev.py` to do projection from BEV to front view.
3. Use `python task2_load.py` to collect RGB, depth images and ground truth poses for scene reconstruction. There is one argument that can be pass into:
-f to select floor [0, 1]
4. Use `python reconstruction.py` to do scene reconstruction. There are some arguments that can be pass into:
-i to select ICP methods [open3d, own], -v to set voxel size, -f to select floor [0, 1]
