import h5py
from pathos import multiprocessing
import numpy as np
import time
import os
import glob 
import constructive_tree.generate_constructive_tree as Gen
from constructive_tree.primitive_pymesh import Primitive


def gen_ply(save_dir, gen_num, leaf_num, workers=1, count_start=0):
    if workers <= 1:
        for i in range(count_start, count_start+gen_num):
            save_name  = os.path.join(save_dir, "{}_{}.ply".format(leaf_num, i))
            Gen.random_csg_tree_gen_and_save(save_name,leaf_num)
    else:
        pool = multiprocessing.Pool(processes=workers)
        for i in range(count_start, count_start+gen_num):
            save_name  = os.path.join(save_dir, "{}_{}.ply".format(leaf_num, i))
            pool.apply_async(Gen.random_csg_tree_gen_and_save, (save_name,leaf_num,  ))
        pool.close() 
        pool.join() 
        

def sample_ply_file(filename, num_points):
    mesh = Primitive.load(filename)
    try:
        pts, sem, ins, norm = mesh.sample_points(num_points, return_attributes=['label', 'source', 'normal'])
        return pts, sem, ins, norm
    except Exception as e:
        print(e, filename)
        pts = mesh.sample_points(num_points)
        sem, ins, nrom = None, None, None
        pts = pts[0]
    return pts, sem, ins, norm 

def ply2h5(ply_list, h5_file,  num_points=8192, face_tracking=True):
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('data', (len(ply_list), num_points, 3), dtype='f')
        hf.create_dataset('p_num', (len(ply_list), ), dtype='i')
        hf.create_dataset('dof', (len(ply_list), ), dtype='i')
        
        for i, ply in enumerate(ply_list):
            if i % 2000 == 0:
                print('{}/{}'.format(i, len(ply_list)))
            file_name =  os.path.splitext(os.path.basename(ply))[0]
            p_num, index, dof = tuple([int(_) for _ in file_name.split('_')])
            data, sem_label, ins_label, normal = sample_ply_file(ply,num_points)
            hf['data'][i, ...] = data[:]
            if face_tracking:
                if i == 0:
                    print('Using face tracking...')
                    hf.create_dataset('ins_label', (len(ply_list), num_points, ), dtype='i')
                    hf.create_dataset('sem_label', (len(ply_list), num_points, ), dtype='i')
                    hf.create_dataset('normal', (len(ply_list), num_points, 3), dtype='f')
                hf['ins_label'][i, ...] = np.squeeze(ins_label.astype(np.int))
                hf['sem_label'][i, ...] = np.squeeze(sem_label.astype(np.int))
                hf['normal'][i, ...] = np.squeeze(normal.astype(np.float))
            hf['p_num'][i] = p_num
            hf['dof'][i] = dof
    