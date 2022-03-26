# Primtive3D
This REPO is for CVPR2022 Paper: Primitive3D: 3D Object Dataset Synthesis from Randomly Assembled Primitives
<div align="center">
	<img src="compare.png" alt="Editor" width="500">
</div>
## Installation
All experiments have been tested on Python 3.6 and CUDA10.2 environment.
### Requirements for Data Generation
**Install PyMesh：**
PyMesh can be install with the '.whl' files in the latest released [version 3.0](https://github.com/PyMesh/PyMesh/releases/tag/v0.3).

**Other dependencies：**
Other depent packages can be installed by the following command
```bash
pip install -r datagen_requirements.txt
```

## Run Details
### Primitive3D Generation
To generate Primitive3D dataset in '.h5' format, please run:
```bash
python gen_primitive3d.py
``` 
The dataset generation contains two step: 
* generate mesh-based objects with '.ply' formats in `./data/primitive3d_ply`. 
* generate point cloud with '.h5' format in `./data/primitive3d.h5`. 

User can change the default path by modify `gen_primitive3d.py`. Other modification can be done to determine the  statistics of dataset generation in this file. The default generated dataset is a small-scale dataset.


## Download Link
