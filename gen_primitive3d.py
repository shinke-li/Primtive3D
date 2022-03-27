import glob 
from constructive_tree import gen_ply as gen

def save_ply(save_dir, gen_dict, workers=1, tracking_face=True):
    gen.Gen.pp.ENGINE = 'igl' if tracking_face else 'corefinement'
    total_num = 0
    for k, v in gen_dict.items():
        gen.gen_ply(save_dir, v, k, workers=workers, count_start=total_num)
        total_num += v
        
def save_points(save_h5file, load_dir,num_points=8192, tracking_face=True):
    ply_list = glob.glob(load_dir + '/' + '*.ply')
    gen.ply2h5(ply_list,save_h5file,num_points=num_points, face_tracking=tracking_face)
    
if __name__ == '__main__':
    import os
    gen_dict = {1:2000, 2:6000, 3:12000, 4:30000, 5:50000, 6:50000}
    
    tracking_face = True
    ply_save_dir = './data/primitive3d_ply'
    h5_save_file = './data/primitive3d.h5'
    
    os.makedirs(ply_save_dir, exist_ok=True)
    save_ply(ply_save_dir, gen_dict, workers=16, tracking_face=tracking_face) #workers 16
    save_points(h5_save_file, ply_save_dir, tracking_face=tracking_face)