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
        polys = np.asarray(bm.faces)
        logging.info(f"Blender-Preprocessing took {stop - start}")

        # ================================================
        # Prepare textures
        # ================================================
        if (colorize == 'UVTEX'):
            try:
                if (o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
                if (uvtexnode is None):
                    raise Exception("Cannot find active image texture in active material")
                uvimage = uvtexnode.image
                if (uvimage is None):
                    raise Exception("Cannot find active image texture with loaded image in active material")
                uvimage.update()
                uvarray = np.asarray(uvimage.pixels)
                uvarray = uvarray.reshape((uvimage.size[1], uvimage.size[0], 4))
                uvlayer = bm.loops.layers.uv.active
                if (uvlayer is None):
                    raise Exception("Cannot find active UV layout")
            except Exception as e:
                raise Exception(str(e))

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

        def _apply_texture_color(poly, point):
            def _remap(v, min1, max1, min2, max2, ):
                def clamp(v, vmin, vmax):
                    if (vmax <= vmin):
                        raise ValueError("Maximum value is smaller than or equal to minimum.")
                    if (v <= vmin):
                        return vmin
                    if (v >= vmax):
                        return vmax
                    return v

                def normalize(v, vmin, vmax):
                    return (v - vmin) / (vmax - vmin)

                def interpolate(nv, vmin, vmax):
                    return vmin + (vmax - vmin) * nv

                if (max1 - min1 == 0):
                    # handle zero division when min1 = max1
                    return min2

                r = interpolate(normalize(v, min1, max1), min2, max2)
                return r

            if (colorize == 'UVTEX'):
                uvtriangle = []
                for l in poly.loops:
                    uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0,)))
                uvpoint = barycentric_transform(point, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(_remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(_remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                return uvarray[y][x][:3].tolist()

        # ================================================
        # Apply texture
        # ================================================
        if (colorize == 'UVTEX'):
            cs = np.array(list(map(_apply_texture_color, polys[weighted_random_indices], vs)))
        else:
            cs = np.full_like(vs, (1.0, 0.0, 0.0,) if not constant_color else constant_color)

        ns = []
        ns = np.array(ns, dtype=np.float)

        return vs, ns, cs
