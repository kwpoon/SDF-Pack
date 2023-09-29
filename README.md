<h2 align = "center"> SDF-Pack: Towards Compact Bin Packing with Signed-Distance-Field Minimization</center></h2>
<h4 align = "center">Jia-Hui Pan<sup>1</sup>, Ka-Hei Hui<sup>1</sup>, Xiaojie Gao<sup>1,3</sup>, Shize Zhu<sup>3</sup>, Yun-Hui Liu<sup>2,3</sup>, and Chi-Wing Fu<sup>1</sup></h4>
<h4 align = "center"> <sup>1</sup>Department of Computer Science and Engineering</center>, <sup>2</sup>Department of Mechanical and Automation Engineering, </h4>
<h4 align = "center"> The Chinese University of Hong Kong.</center></h4>
<h4 align = "center"> <sup>3</sup>Hong Kong Centre for Logistics Robotics</center></h4>.


### Introduction
This repository is for our IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2023 paper 'SDF-Pack: Towards Compact Bin Packing with Signed-Distance-Field Minimization'. In this paper, we present a novel packing method named SDF-Pack which leverages the truncated signed distance field to model the container’s geometric condition. We also developed the SDF-minimization heuristic to effectively evaluate spatial compactness and find compact object placements. 

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./figures/framework.png alt="">
    <br>
</div>

This repository contains the key codes for constructing the signed distance field (SDF) based on the container's top-down heightmap, the key implementation of the SDF-Minimization packing heuristics, and we keep updating it.

## Todo List

* [X] ~~SDF-Minimization heuristics~~
* [X] ~~GPU-based SDF construction for the container~~
* [ ] Experimental dataset
* [ ] Data preprocessing for novel objects
* [ ] Physical simulator

