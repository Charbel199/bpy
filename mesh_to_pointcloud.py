# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# Copyright (c) 2023 Jakub Uhlik

# This code is meant to use Jakob Uhlik's blender triangle surface sampler which can be found at https://github.com/uhlik/bpy/blob/master/space_view3d_point_cloud_visualizer.py,
# optimize it and add new functionalities to it with the aim to use it purely in code (Not as a blender add-on).

import numpy as np
import statistics
import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion, Color
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
import math
import logging

logging.basicConfig(level=logging.INFO)


class TriangleSurfaceSampler:
    def __init__(self):
        pass

    @staticmethod
    def mesh_to_pointcloud(context, o, num_samples, rnd, colorize=None, constant_color=None,
                           compute_normals=False, exact_number_of_points=False):

        def _normalize(v, vmin, vmax):
            return (v - vmin) / (vmax - vmin)

        def _interpolate(nv, vmin, vmax):
            return vmin + (vmax - vmin) * nv

        def _remap(v, min1, max1, min2, max2, ):
            def clamp(v, vmin, vmax):
                if (vmax <= vmin):
                    raise ValueError("Maximum value is smaller than or equal to minimum.")
                if (v <= vmin):
                    return vmin
                if (v >= vmax):
                    return vmax
                return v

            if (max1 - min1 == 0):
                # handle zero division when min1 = max1
                return min2

            r = _interpolate(_normalize(v, min1, max1), min2, max2)
            return r

        def _random_point_in_triangle(a, b, c, ):
            r1 = rnd.random()
            r2 = rnd.random()
            p = (1 - math.sqrt(r1)) * a + (math.sqrt(r1) * (1 - r2)) * b + (math.sqrt(r1) * r2) * c
            return p

        def _generate(poly, vs, ns, cs, override_num=None, ):
            ps = poly.verts
            tri = (ps[0].co, ps[1].co, ps[2].co)
            # if num is 0, it can happen when mesh has large and very small polygons, increase number of samples and eventually all polygons gets covered
            num = int(round(_remap(poly.calc_area(), area_min, area_max, min_ppf, max_ppf)))
            if (override_num is not None):
                num = override_num
            for i in range(num):
                # Generate POINTS
                v = _random_point_in_triangle(*tri)
                vs.append(v.to_tuple())

                # Generate NORMALS
                if compute_normals:
                    if (poly.smooth):
                        a = poly.verts[0].normal
                        b = poly.verts[1].normal
                        c = poly.verts[2].normal
                        nws = poly_3d_calc([a, b, c, ], v)

                        nx = a.x * nws[0] + b.x * nws[1] + c.x * nws[2]
                        ny = a.y * nws[0] + b.y * nws[1] + c.y * nws[2]
                        nz = a.z * nws[0] + b.z * nws[1] + c.z * nws[2]
                        normal = Vector((nx, ny, nz)).normalized()
                        ns.append(normal.to_tuple())
                    else:
                        ns.append(poly.normal.to_tuple())

                # Generate COLORS
                if (colorize is None):
                    cs.append((1.0, 0.0, 0.0,))
                elif (colorize == 'CONSTANT'):
                    cs.append(constant_color)
                elif (colorize == 'VCOLS'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    ac = poly.loops[0][col_layer][:3]
                    bc = poly.loops[1][col_layer][:3]
                    cc = poly.loops[2][col_layer][:3]
                    r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                    g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                    b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                    cs.append((r, g, b,))
                elif (colorize == 'UVTEX'):
                    uvtriangle = []
                    for l in poly.loops:
                        uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0,)))
                    uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co,
                                                    *uvtriangle, )
                    w, h = uvimage.size
                    # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                    x = int(round(_remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                    y = int(round(_remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                    cs.append(tuple(uvarray[y][x][:3].tolist()))
                elif (colorize == 'GROUP_MONO'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    cs.append((m, m, m,))
                elif (colorize == 'GROUP_COLOR'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    hue = _remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                    c = Color()
                    c.hsv = (hue, 1.0, 1.0,)
                    cs.append((c.r, c.g, c.b,))



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

        areas = tuple([p.calc_area() for p in bm.faces])
        if (sum(areas) == 0.0):
            raise Exception("Mesh surface area is zero")
        area_min = min(areas)
        area_max = max(areas)
        avg_ppf = num_samples / len(areas)
        area_med = statistics.median(areas)
        nums = []
        for p in bm.faces:
            r = p.calc_area() / area_med
            nums.append(avg_ppf * r)

        max_ppf = max(nums)
        min_ppf = min(nums)

        vs = []
        ns = []
        cs = []

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
        if (colorize == 'VCOLS'):
            try:
                col_layer = bm.loops.layers.color.active
                if (col_layer is None):
                    raise Exception()
            except Exception:
                raise Exception("Cannot find active vertex colors")
        if (colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if (group_layer is None):
                    raise Exception()
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")

        for poly in bm.faces:
            _generate(poly, vs, ns, cs, )

        if (exact_number_of_points):
            if (len(vs) < num_samples):
                while (len(vs) < num_samples):
                    # generate one sample in random face until full
                    poly = bm.faces[rnd.randrange(len(bm.faces))]
                    _generate(poly, vs, ns, cs, override_num=1, )
            if (len(vs) > num_samples):
                a = np.concatenate((vs, ns, cs), axis=1, )
                np.random.shuffle(a)
                a = a[:num_samples]
                # vs = np.column_stack((a[:, 0], a[:, 1], a[:, 2], ))
                # ns = np.column_stack((a[:, 3], a[:, 4], a[:, 5], ))
                # cs = np.column_stack((a[:, 6], a[:, 7], a[:, 8], ))
                vs = a[:, :3]
                ns = a[:, 3:6]
                cs = a[:, 6:]

        if (len(vs) == 0):
            raise Exception("No points generated, increase number of points or decrease minimal distance")

        return vs, ns, cs
