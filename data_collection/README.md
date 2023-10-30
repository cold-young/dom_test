# Data Collection module for srl

* Chanyoung Ahn (cold-young)

* data_collection.py
   - You can collecting deformable fruit point cloud data.
   
   - Examples
   ```bash
   python data_collection.py --object=tofu --num=10 --gravity --extract_indices
   ```

    - Can use various options
   ```
   headless, 
   data_num, 
   norm_pcd, 
   object, 
   gravity, 
   extract_indices, 
   vis_pcd
   ```


-----------------------------------------------

* "sim_to_npy.py"
Input  : Sim_data.json
OUTPUT : Oring.npy (Full object's point cloud)

* "np_to_xyzn.py"  
Input  : Oring.npy, Oring_Indices.npy
OUTPUT : oring/X.xyzn
- From full-size PC to exterior vertices
- create .xyzn file

* "np_to_xyzn_visualziation.py"
- if you want to check visually "np_to_xyzn.py", use this file
- Please change "time" on line 72

* "xyzn_check.py"
Input : oring/X.xyzn
- time-seriese check
- press "a" to animate your figure

