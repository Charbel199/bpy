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
from itertools import repeat

logging.basicConfig(level=logging.INFO)


class TriangleSurfaceSampler:
    def __init__(self):
        pass

    @staticmethod
    def mesh_to_pointcloud(context, o, num_samples, rnd, colorize=None, constant_color=None,
                           compute_normals=False, exact_number_of_points=False):

        num_samples = int(num_samples)

        # ================================================
        # Prepare Mesh
        # ================================================
        start = time.time_ns() / (10 ** 9)
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
        logging.info(f"Blender-Preprocessing took {stop - start}")

        polys = np.asarray(bm.faces)

        # ================================================
        # Prepare vertices
        # ================================================
        def _get_vertex(poly, index=0):
            return poly.verts[index].co

        v1_xyz = np.array(list(map(_get_vertex, polys, repeat(0))))
        v2_xyz = np.array(list(map(_get_vertex, polys, repeat(1))))
        v3_xyz = np.array(list(map(_get_vertex, polys, repeat(2))))

        # ================================================
        # Prepare areas
        # ================================================
        def _get_area(poly):
            return poly.calc_area()

        areas = np.array(list(map(_get_area, polys)))


        # ================================================
        # Prepare weighted random indices
        # ================================================
        probabilities = areas / sum(areas)
        weighted_random_indices = np.random.choice(range(len(areas)), size=num_samples, p=probabilities)

        # ================================================
        # Fetch random polygons' vertices
        # ================================================
        v1_xyz = v1_xyz[weighted_random_indices]
        v2_xyz = v2_xyz[weighted_random_indices]
        v3_xyz = v3_xyz[weighted_random_indices]

        # ================================================
        # Create points
        # ================================================
        u = np.random.rand(num_samples, 1)
        v = np.random.rand(num_samples, 1)
        is_a_problem = u + v > 1
        u[is_a_problem] = 1 - u[is_a_problem]
        v[is_a_problem] = 1 - v[is_a_problem]
        w = 1 - (u + v)
        vs = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)
        vs = vs.astype(np.float)

        ns = []
        cs = np.full_like(vs, (1.0, 0.0, 0.0,))
        ns = np.array(ns, dtype=np.float)
        cs = np.array(cs, dtype=np.float)

        return vs, ns, cs
