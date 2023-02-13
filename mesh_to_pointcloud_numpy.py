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
import cv2

logging.basicConfig(level=logging.INFO)


class TriangleSurfaceSampler:
    def __init__(self):
        pass

    @staticmethod
    def mesh_to_pointcloud(context, o, num_samples, rnd, colorize=None, constant_color=None,
                           compute_normals=False, exact_number_of_points=False):

        num_samples = int(num_samples)
        conversion_summary = []

        # ================================================
        # Prepare Mesh
        # ================================================
        start = time.time_ns() / (10 ** 9)
        depsgraph = context.evaluated_depsgraph_get()
        if o.modifiers:
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
        if len(bm.faces) == 0:
            raise Exception("Mesh has no faces")
        stop = time.time_ns() / (10 ** 9)
        polys = np.asarray(bm.faces)
        logging.info("Done Prepare Mesh")
        conversion_summary.append(f"Preparing Mesh: {stop - start}")

        # ================================================
        # Prepare textures
        # ================================================
        start = time.time_ns() / (10 ** 9)
        if colorize == 'UVTEX':
            try:
                if o.active_material is None:
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
                if uvtexnode is None:
                    raise Exception("Cannot find active image texture in active material")
                uvimage = uvtexnode.image
                if uvimage is None:
                    raise Exception("Cannot find active image texture with loaded image in active material")
                uvimage.update()
                file_path = uvimage.filepath
                start1 = time.time_ns() / (10 ** 9)
                uvarray = cv2.imread(file_path)
                uvarray = uvarray / 255.0
                uvarray = uvarray[..., ::-1]
                uvarray = np.concatenate((uvarray, np.ones((uvarray.shape[0], uvarray.shape[1], 1))), axis=2)
                logging.info(f"Working with texture {file_path} with shape {uvarray.shape}")
                stop1 = time.time_ns() / (10 ** 9)
                conversion_summary.append(f"Texture to numpy array: {stop1 - start1}")
                uvlayer = bm.loops.layers.uv.active
                if uvlayer is None:
                    raise Exception("Cannot find active UV layout")
            except Exception as e:
                logging.info(f"Error while loading texture {str(e)}")
                raise Exception(str(e))
        logging.info("Done Prepare textures")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Prepare textures: {stop - start}")

        # ================================================
        # Prepare vertices
        # ================================================
        start = time.time_ns() / (10 ** 9)

        def _get_vertex(poly, index=0):
            return poly.verts[index].co

        v1_xyz = np.array(list(map(_get_vertex, polys, repeat(0))))
        v2_xyz = np.array(list(map(_get_vertex, polys, repeat(1))))
        v3_xyz = np.array(list(map(_get_vertex, polys, repeat(2))))
        logging.info("Done Prepare vertices")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Prepare vertices: {stop - start}")

        # ================================================
        # Prepare areas
        # ================================================
        start = time.time_ns() / (10 ** 9)

        def _get_area(poly):
            return poly.calc_area()

        areas = np.array(list(map(_get_area, polys)))
        logging.info("Done Prepare areas")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Prepare areas: {stop - start}")

        # ================================================
        # Prepare weighted random indices
        # ================================================
        start = time.time_ns() / (10 ** 9)
        probabilities = areas / sum(areas)
        weighted_random_indices = np.random.choice(range(len(areas)), size=num_samples, p=probabilities)
        logging.info("Done Prepare weighted random indices")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Prepare weighted random indices: {stop - start}")

        # ================================================
        # Fetch random polygons' vertices
        # ================================================
        start = time.time_ns() / (10 ** 9)
        v1_xyz = v1_xyz[weighted_random_indices]
        v2_xyz = v2_xyz[weighted_random_indices]
        v3_xyz = v3_xyz[weighted_random_indices]
        logging.info("Done Fetch random polygons' vertices")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Fetch random polygons' vertices: {stop - start}")

        # ================================================
        # Create points
        # ================================================
        start = time.time_ns() / (10 ** 9)
        u = np.random.rand(num_samples, 1)
        v = np.random.rand(num_samples, 1)
        is_a_problem = u + v > 1
        u[is_a_problem] = 1 - u[is_a_problem]
        v[is_a_problem] = 1 - v[is_a_problem]
        w = 1 - (u + v)
        vs = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)
        logging.info("Done Create points")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Create points: {stop - start}")

        # ================================================
        # Apply texture
        # ================================================
        start = time.time_ns() / (10 ** 9)

        def _apply_texture_color(poly, point):
            def _remap(v, min1, max1, min2, max2, ):
                def clamp(v, vmin, vmax):
                    if vmax <= vmin:
                        raise ValueError("Maximum value is smaller than or equal to minimum.")
                    if v <= vmin:
                        return vmin
                    if v >= vmax:
                        return vmax
                    return v

                def normalize(v, vmin, vmax):
                    return (v - vmin) / (vmax - vmin)

                def interpolate(nv, vmin, vmax):
                    return vmin + (vmax - vmin) * nv

                if max1 - min1 == 0:
                    # handle zero division when min1 = max1
                    return min2

                r = interpolate(normalize(v, min1, max1), min2, max2)
                return r

            if colorize == 'UVTEX':
                uvtriangle = []
                for l in poly.loops:
                    uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0,)))
                uvpoint = barycentric_transform(point, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co,
                                                *uvtriangle, )
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(_remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(_remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                return uvarray[y][x][:3].tolist()

        if colorize == 'UVTEX':
            cs = np.array(list(map(_apply_texture_color, polys[weighted_random_indices], vs)))
        else:
            cs = np.full_like(vs, (1.0, 0.0, 0.0,) if not constant_color else constant_color)
        logging.info("Done Apply texture")
        stop = time.time_ns() / (10 ** 9)
        conversion_summary.append(f"Apply texture: {stop - start}")

        # ================================================
        # Get normals
        # ================================================
        ns = []
        ns = np.array(ns)
        logging.info(f"\n=========================")
        logging.info(f"SUMMARY:")
        logging.info("\n".join(conversion_summary))
        logging.info(f"=========================\n")
        return vs, ns, cs
