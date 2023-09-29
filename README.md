<h2 align = "center"> SDF-Pack: Towards Compact Bin Packing with Signed-Distance-Field Minimization</center></h2>
<h4 align = "center">Jia-Hui Pan<sup>1</sup>, Ka-Hei Hui<sup>1</sup>, Xiaojie Gao<sup>1,3</sup>, Shize Zhu<sup>3</sup>, Yun-Hui Liu<sup>2,3</sup>, and Chi-Wing Fu<sup>1</sup></h4>
<h4 align = "center"> <sup>1</sup>Department of Computer Science and Engineering</center>, <sup>2</sup>Department of Mechanical and Automation Engineering, </h4>
<h4 align = "center"> The Chinese University of Hong Kong.</center></h4>
<h4 align = "center"> <sup>3</sup>Hong Kong Centre for Logistics Robotics</center></h4>.


### Introduction
This repository is for our IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2023 paper *'SDF-Pack: Towards Compact Bin Packing with Signed-Distance-Field Minimization'*. In this paper, we present a novel packing method named SDF-Pack which leverages the truncated signed distance field to model the container’s geometric condition. We also developed the SDF-minimization heuristic to effectively evaluate spatial compactness and find compact object placements. 

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./figures/framework.png alt="">
    <br>
</div>

*The current version contains the key implementation for the signed distance field (SDF) construction given the container's top-down heightmap and that of the SDF-Minimization packing heuristics. We are still updating the repository.*

### Dataset Preparation
We performed experiments on 1000 packing sequences of 96 types of objects collected from the YCB dataset and the Rutgers APC RGB-D dataset. Please download the processed dataset from [Google Drive](https://drive.google.com/file/d/1i2iPqhWSmGWMJC3wa9Y_fVD3HyuklFAO/view?usp=sharing) and extract the files in the folder `./dataset/`. The object IDs forming the packing sequences can be found at `1000_packing_sequences_of_80_objects.npy`.
```
|-- 1000_packing_sequences_of_80_objects.npy
|-- dataset  
|   |-- our_oriented_dataset
|   |   |-- 00000003_072-c_toy_airplane-processed.ply
|   |   |...
|   |-- our_oriented_decomp
|   |   |-- 00000003_072-c_toy_airplane-processed.obj
|   |   |...
|   |-- our_oriented_occs
|   |   |-- 00002777_cheezit_big_original-processed_objocc.npy
|   |   |-- 00002777_cheezit_big_original-processed_depth.npy
|   |   |...
```
The subfolder `./dataset/our_oriented_dataset/` contains the object meshes that are simplified and processed to be watertight. These meshes are further processed through V-HACD convex decomposition for collision simulation, and the processed collision models are presented in the folder `./dataset/our_oriented_decomp/`. We also provide the voxelization results of the objects in `./dataset/our_oriented_occs/`. 

### Key Implementation
Our key implementation of the GPU-based SDF construction and the SDF-Minimization packing heuristics can be found at `v11_heuristic_packing.py`.

## Install

* Environment
  ```
  conda env create -f environment.yml
  conda activate sdf_pack
  ```

### Todo List

* [X] ~~SDF-Minimization heuristics~~
* [X] ~~GPU-based SDF construction for the container~~
* [X] ~~Experimental dataset~~
* [ ] Data preprocessing for novel objects
* [ ] Physical simulator

