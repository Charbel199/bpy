import numpy as np
import statistics
import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion, Color
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
import math
import logging
import time

logging.basicConfig(level=logging.INFO)


class TriangleSurfaceSampler:
    def __init__(self):
        pass

    @staticmethod
    def mesh_to_pointcloud(context, o, num_samples, rnd, colorize=None, constant_color=None,
                           compute_normals=False, exact_number_of_points=False):

        vs_time = []
        pre_vs_time = []
        num_samples = int(num_samples)
        start = time.time_ns() / (10 ** 9)
        # Mesh stuff
        depsgraph = context.evaluated_depsgraph_get()
        if (o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if (len(bm.faces) == 0):
            raise Exception("Mesh has no faces")
        stop = time.time_ns() / (10 ** 9)
        logging.info(f"Blender-Preprocessing took {stop-start}")


        # Areas and weighted indices
        areas = [p.calc_area() for p in bm.faces]
        areas = np.asarray(areas)
        probabilities = areas/sum(areas)
        weighted_random_indices = np.random.choice(range(len(areas)), size=num_samples, p=probabilities)
        v1_xyz = []
        v2_xyz = []
        v3_xyz = []
        for poly in bm.faces:
            v1_xyz.append(poly.verts[0].co)
            v2_xyz.append(poly.verts[1].co)
            v3_xyz.append(poly.verts[2].co)

        v1_xyz = np.asarray(v1_xyz)
        v2_xyz = np.asarray(v2_xyz)
        v3_xyz = np.asarray(v3_xyz)

        v1_xyz = v1_xyz[weighted_random_indices]
        v2_xyz = v2_xyz[weighted_random_indices]
        v3_xyz = v3_xyz[weighted_random_indices]

        u = np.random.rand(num_samples,1)
        v = np.random.rand(num_samples,1)
        is_a_problem = u+v>1
        u[is_a_problem] = 1 - u[is_a_problem]
        v[is_a_problem] = 1 - v[is_a_problem]
        w = 1 - (u+v)

        vs = (v1_xyz * u) + (v2_xyz * v) + (w*v3_xyz)
        vs = vs.astype(np.float)
        ns = []
        cs = np.full_like(vs, (1.0, 0.0, 0.0,))
        ns = np.array(ns, dtype=np.float)
        cs = np.array(cs, dtype=np.float)

        return vs, ns, cs
