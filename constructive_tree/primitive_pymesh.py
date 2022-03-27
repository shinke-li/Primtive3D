import open3d as o3d

import numpy as np
import random
from scipy.spatial.distance import cdist
import time
try:
    import pymesh
except Exception as e:
    print('pymesh cannot be imported')

ENGINE = 'corefinement'#'igl'
PRINT = False

def myprint(*args, **kwargs):
    if PRINT:
        print(*args, **kwargs)
    
def gaussian_dist(mean, sigma, clip_r=None, attempt=100):
    # clip_r: None: sample @ [(mean - sigma), (mean + sigma)] ~ 0.7
    # clip_r: float: sample @ [(mean - clip_r * sigma), (mean + clip_r * sigma)] ~ 1
    if clip_r is None:
        return random.gauss(0, 1) * sigma + mean
    else:
        ns_list = np.random.randn(attempt)
        samples = ns_list * sigma + mean
        truncated_locs =  np.argwhere((samples >= mean - sigma * clip_r) & 
                                     (samples <= mean + sigma * clip_r))
        assert len(truncated_locs) > 0, 'No samples within {} attempts'.format(attempt)
        return samples[np.min(truncated_locs)]

def uniform_dist(high, low):
    return random.uniform(low, high)


def face_mapping_pymesh(m1, m2, outputm):
    tic = time.time()

    v1 = m1.vertices
    v2 = m2.vertices
    f1 = m1.faces 
    f2 = m2.faces + v1.shape[0]
    
    m = pymesh.meshio.form_mesh(np.vstack([v1, v2]), np.vstack([f1, f2]))
    
    m.add_attribute("source_face")
    m.set_attribute("source_face", np.arange(f1.shape[0]+f2.shape[0]))
    
    pymesh.map_face_attribute(m, outputm, 'source_face')
    source_face = outputm.get_attribute('source_face')
    return source_face

    
def mesh_boolean(mesh1, mesh2, operation, engine):
    if engine == 'igl':
        mesh = pymesh.boolean(mesh1, mesh2, operation=operation, engine=engine)
    else:
        tic = time.time()
        mesh = pymesh.boolean(mesh1, mesh2, operation=operation, engine=engine)
        myprint(time.time() - tic, 'pure_boolean time')
        source_face = face_mapping_pymesh(mesh1, mesh2, mesh)
        
        mesh.add_attribute("source_face")
        mesh.set_attribute("source_face", source_face)
       
    return mesh
    
        
class Primitive(object):
    #instance label: face_source
    #semantic label: face_label
    
    DEFAULT_ATTRIBUTES = ['face_area', 'face_centroid', 'face_normal', 'vertex_normal', 'face_index','vertex_area','face_index','vertex_index' ]
    EXTRA_ATTRIBUTES = [ 'face_source','face_color',  'vertex_color',  'vertex_source', 'face_label', 'vertex_label']
    ATTRIBUTES_NAME = DEFAULT_ATTRIBUTES + EXTRA_ATTRIBUTES
    LABEL = 0
    RDOF = -1e5
    DOF = -1e5
    def __init__(self, vertex=None, face=None, *args, **kwargs):
        self.mesh = None
        if (vertex is None) or (face is None): #deuplicate one
            if len(args)>0 and isinstance(args[0], pymesh.Mesh):
                mesh = args[0]
            elif "mesh" in kwargs.keys() and isinstance(kwargs["mesh"], pymesh.Mesh):
                mesh = kwargs["mesh"]
            else:
                raise TypeError("missing required positional argument")
            self.mesh = mesh
        else:# create one from elements
            self.mesh = pymesh.meshio.form_mesh(vertex, face)
            for attr in self.ATTRIBUTES_NAME:
                if attr in kwargs.keys():self.__init_attributes(attr, kwargs[attr])
        
        
        self.__init_attributes('face_label', np.ones(self.mesh.num_faces, ) * self.LABEL)
        self.__init_attributes('face_source', np.zeros(self.mesh.num_faces, ))
    
    def __get_mesh_attr(self, name):
        if name.startswith('face'):
            return self.mesh.get_face_attribute(name)
        elif name.startswith('vertex'):
            return self.mesh.get_vertex_attribute(name)
        
    def __getattr__(self, name): # after get attribute
        if self.mesh.has_attribute(name):
            return self.__get_mesh_attr(name)
        elif name in ["faces", "vertices", "bbox"]:
            return getattr(self.mesh, name)
        elif name in self.DEFAULT_ATTRIBUTES:
            self.mesh.add_attribute(name)
            return self.__get_mesh_attr(name)
        else: 
            raise AttributeError("pymesh.Mesh does not have attribute \"{}\"".format(name))
            
    def __init_attributes(self, attr, vals):
        mesh_attr = getattr(self, attr, None)
        if mesh_attr is None: 
            self.mesh.add_attribute(attr)
            self.mesh.set_attribute(attr, vals)
    
    @staticmethod
    def get_rotation_matrix(*args):
        if len(args) >= 2:
            return Primitive._get_rotation_matrix_from_axis_angle(args[0],args[1])
        elif len(args) == 1:
            return Primitive._get_rotation_matrix_from_euler_angles(angles)
        else:
            assert False 
            
    @staticmethod
    def _get_rotation_matrix_from_axis_angle(axis,angle):
        rot = pymesh.Quaternion.fromAxisAngle(axis, angle)
        return rot.to_matrix()
    
    @staticmethod
    def _get_rotation_matrix_from_euler_angles(angles):
        rot = open3d.geometry.get_rotation_matrix_from_xyz(angles)
        return rot
    
    def get_attributes_dict(self):
        d = {}
        for attr in self.ATTRIBUTES_NAME:
            if self.mesh.has_attribute(attr):
                d.update({attr:self.mesh.get_attribute(attr)})
        return d 

    def rigid_transform(self, axis=[1.,0.,0.],angle=0., offset=[0.,0.,0.]):
        rot = self.get_rotation_matrix(axis, angle)
        center = self.get_center()
        vertices = self.mesh.vertices
        vertices = np.dot(rot, (vertices - center).T).T + center + offset
        d = self.get_attributes_dict()
        self.__init__(vertex=np.squeeze(vertices), face=self.mesh.faces, **d)
    
    
    def get_center(self, center_type="mean_vertex"):#"mean_vertex"
        assert center_type in ['bbox', "mean_vertex" ], '{} is not a valid center type'.format(center_type)
        if center_type == "bbox":
            bbox = self.mesh.bbox
            return 0.5 * (bbox[0] + bbox[1])
        elif center_type == "mean_vertex":
            return np.mean(self.mesh.vertices, axis=0) 
    
    def scale(self, scale_factor=1.0):
        vertices = self.mesh.vertices * scale_factor
        d = self.get_attributes_dict()
        self.__init__(vertices, self.mesh.faces, **d)
    
        
    def normalize(self, norm_center=None, center_type="bbox"): #np.array([0.0, 0.0, 0.0]), ):
        # norm_center-0.5 ~ norm_center+0.5 coordinate, 
        # the point located on norm center depends on the center_type
        center = self.get_center(center_type)
        bbox = self.mesh.bbox
        center_min_bound = np.max(center - bbox[0])
        center_max_bound = np.max(bbox[1] - center)   
        half_bound = np.max([center_min_bound, center_max_bound])
        self.scale(0.5 /(half_bound))
        if not(norm_center is None):
            self.rigid_transform(offset=norm_center - self.get_center())

    def  __add__(self, other):
        assert isinstance(other, Primitive), 'Should use {} to perform this operation'.format('Primitive')
        output_mesh = mesh_boolean(self.mesh, other.mesh, operation="union", engine=ENGINE)
        m = self.__assign_label_and_create(output_mesh, other)
        return m
    
    def  __mul__(self, other):
        assert isinstance(other, Primitive), 'Should use {} to perform this operation'.format('Primitive')
        output_mesh = mesh_boolean(self.mesh, other.mesh, operation="intersection",engine=ENGINE)
        return self.__assign_label_and_create(output_mesh, other)
    
    def  __sub__(self, other):
        assert isinstance(other, Primitive), 'Should use {} to perform this operation'.format('Primitive')
        output_mesh = mesh_boolean(self.mesh, other.mesh, operation="difference",engine=ENGINE)
        return self.__assign_label_and_create(output_mesh, other)
    
    def __assign_label_and_create(self, output_mesh, other):
        if True: #ENGINE == 'igl':
            face_indices = output_mesh.get_attribute("source_face").astype(int)
            attr_dict = {}
            for attr_name in self.ATTRIBUTES_NAME:
                if attr_name.startswith('face'):
                    if self.mesh.has_attribute(attr_name) and other.mesh.has_attribute(attr_name):
                        self_attr = self.mesh.get_attribute(attr_name)
                        other_attr = other.mesh.get_attribute(attr_name)
                        if attr_name == 'face_source':
                            other_attr = other_attr + np.max(self_attr) + 1
                        output_attr = np.concatenate([self_attr, other_attr])[face_indices]
                        attr_dict.update({attr_name:output_attr})
            return Primitive(output_mesh.vertices, output_mesh.faces, **attr_dict)
        else:
            return Primitive(output_mesh.vertices, output_mesh.faces, )
    
    def save(self, filename):
        pymesh.meshio.save_mesh(filename, self.mesh, *list(self.get_attributes_dict().keys()))
    
    def save_raw(self, filename):
        pymesh.meshio.save_mesh_raw(filename, self.mesh.vertices, self.mesh.faces)
    
    @staticmethod
    def load(file_name):
        mesh = pymesh.meshio.load_mesh(file_name)
        print(mesh)
        return Primitive(mesh=mesh)
        
    @staticmethod
    def cal_faces_area(vertices, faces):
        x = vertices[faces]
        a = x[:, 0, :] - x[:, 1, :]
        b = x[:, 0, :] - x[:, 2, :]
        cross = np.cross(a, b)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def sample_points(self, num_points, return_attributes=[]):
        assert isinstance(return_attributes, list), 'input argument \'return_attributes\' should be list not other type.'
        attr_values = []
        for attr in return_attributes:
            attr_values.append(getattr(self, 'face_' + attr, None))
            assert not (attr_values[-1] is None), '\'{}\' is not a valid face-based attribute.'.format(attr)
        face_vs = self.mesh.vertices[self.mesh.faces]
        area_p = self.cal_faces_area(self.mesh.vertices, self.mesh.faces)
        area_p = area_p / np.sum(area_p)
        face_indices = np.random.choice(np.arange(self.mesh.num_faces), num_points, p=area_p)
        r1, r2 = np.random.rand(int(num_points)),np.random.rand(int(num_points))
        rd_pos = np.asarray([1-np.sqrt(r1),np.sqrt(r1)*(1-r2), np.sqrt(r1)*r2 ]).T
        pts = np.tile(rd_pos[:,0],(3,1)).T * face_vs[face_indices, 0] + \
              np.tile(rd_pos[:,1],(3,1)).T * face_vs[face_indices, 1] + \
              np.tile(rd_pos[:,2],(3,1)).T * face_vs[face_indices, 2]
        results = [pts] + [a[face_indices] for a in attr_values]
        return tuple(results)

    @staticmethod
    def random(*args, **kwargs):
        return Primitive(*args, **kwargs)
    
    def cal_dof(self):
        return self.RDOF + self.DOF 

class Sphere(Primitive):
    RDOF = 0
    DOF = 1
    LABEL = 1
    GEN_FUNC =o3d.geometry.TriangleMesh.create_sphere
    def __init__(self,  *args, **kwargs):
        super(Sphere, self).__init__( *args, **kwargs)

    @staticmethod
    def random(res=20):
        res = int(uniform_dist(low=res - 5, high=res + 5))
        o3d_mesh = Sphere.GEN_FUNC(radius=0.5, resolution=res)
        o3d_v = np.asarray(o3d_mesh.vertices)
        o3d_f = np.asarray(o3d_mesh.triangles)
        return Sphere(vertex=o3d_v,face=o3d_f)
    

    
    
class Cylinder(Primitive):
    RDOF = 2
    DOF = 2
    GEN_FUNC = o3d.geometry.TriangleMesh.create_cylinder
    RATIO_MAX= 5 
    LABEL = 2
    def __init__(self,*args, **kwargs):
        super(Cylinder, self).__init__(*args, **kwargs)
    
    @staticmethod
    def random(resolution=25, split=4):
        low, high = np.arctan(1.0 / Cylinder.RATIO_MAX), np.arctan(Cylinder.RATIO_MAX)
        theta = uniform_dist(low, high)
        h, d = np.sin(theta), np.cos(theta)

        res = uniform_dist(low=resolution - 5, high=resolution + 5)
        o3d_mesh =  Cylinder.GEN_FUNC(radius= d / 2.0, 
                                      height=h, 
                                      resolution=int(res), 
                                      split=split)
        o3d_v = np.asarray(o3d_mesh.vertices)
        o3d_f = np.asarray(o3d_mesh.triangles)
        return Cylinder(vertex=o3d_v,face=o3d_f)
    


class Box(Primitive):
    RDOF = 3
    DOF = 3
    LABEL = 3
    GEN_FUNC = o3d.geometry.TriangleMesh.create_box
    RATIO_MAX = 5
    def __init__(self, *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        
    @staticmethod
    def random():
        low, high = np.arctan(1.0 / Box.RATIO_MAX), np.arctan(Box.RATIO_MAX)
        phi = uniform_dist(low, high)
        s, c = np.sin(phi), np.cos(phi)
        low, high = np.arctan(1.0 / Box.RATIO_MAX / min(s, c)), \
                    np.arctan(Box.RATIO_MAX / max(s ,c))
        theta = uniform_dist(low, high)
        w = np.sin(theta) * np.cos(phi)
        h = np.sin(theta) * np.sin(phi)
        d = np.cos(theta)
        
        o3d_mesh = Box.GEN_FUNC(
                    width=w,
                    height=h,
                    depth=d)
        o3d_mesh.translate(-1 * np.asarray([w,h, d]) / 2)
        o3d_v = np.asarray(o3d_mesh.vertices)
        o3d_f = np.asarray(o3d_mesh.triangles)
        return Box(vertex=o3d_v,face=o3d_f)
    


class Cone(Primitive):
    #0.25 * h
    RDOF = 2
    DOF = 2
    LABEL = 5
    GEN_FUNC = o3d.geometry.TriangleMesh.create_cone
    RATIO_MAX = 5
    def __init__(self, *args, **kwargs):
        super(Cone, self).__init__(*args, **kwargs)
        
    @staticmethod
    def random( resolution=20, split=2):
        low, high = np.arctan(1.0 / Cone.RATIO_MAX), np.arctan(Cone.RATIO_MAX)
        theta = uniform_dist(low, high)
        h, d = np.sin(theta), np.cos(theta)
        eta = h / d
        h_r = min(2 * eta * np.sqrt(1 / (4 + eta**2)), 2.0 / 3.0)
        h, d = h_r, h_r / eta
        res = uniform_dist(low=resolution - 5, high=resolution + 5)

        o3d_mesh = Cone.GEN_FUNC(
                        radius=d / 2.0, 
                        height=h, 
                        resolution=int(res), 
                        split=split)
        
        o3d_mesh.translate(-1 * np.asarray([0,0, h * 0.25]))
        o3d_v = np.asarray(o3d_mesh.vertices)
        o3d_f = np.asarray(o3d_mesh.triangles)
        return Cone(vertex=o3d_v,face=o3d_f)

    
class Torus(Primitive):
    RDOF = 2
    DOF = 3
    LABEL = 4
    GEN_FUNC = o3d.geometry.TriangleMesh.create_torus
    RATIO_MAX = 7
    def __init__(self, *args, **kwargs):
        super(Torus, self).__init__(*args, **kwargs)
        self.DOF = 1 #resolution
    
    @staticmethod
    def random( radial_resolution=30, tubular_resolution=20):
        eta = uniform_dist(low=1.0 / Torus.RATIO_MAX, high= 0.6) #hd_ratio = h / d
        torus_radius = 1.0 / (1 + eta)
        tube_radius = torus_radius * eta
        rad_res = uniform_dist(low=radial_resolution - 10, high=radial_resolution + 10)
        tub_res = uniform_dist(low=tubular_resolution - 5, high=tubular_resolution + 5)
        o3d_mesh = Torus.GEN_FUNC(
                      torus_radius=torus_radius, 
                      tube_radius=tube_radius, 
                      radial_resolution=radial_resolution, 
                      tubular_resolution=tubular_resolution)
        o3d_v = np.asarray(o3d_mesh.vertices)
        o3d_f = np.asarray(o3d_mesh.triangles)
        return Torus(vertex=o3d_v,face=o3d_f)

