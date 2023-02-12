# import numpy as np
import cupy as np
import bmesh
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
        logging.info(f"After areas")
        logging.info(areas.shape)
        sum_of_areas = np.sum(areas)
        probabilities = areas/sum_of_areas
        logging.info(probabilities.shape)
        logging.info(f"After probas")
        weighted_random_indices = np.random.choice(range(len(areas)), size=num_samples, p=probabilities)
        logging.info(f"After weighted_random_indices")
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
        logging.info(f"After v1_xyz")
        v1_xyz = v1_xyz[weighted_random_indices]
        v2_xyz = v2_xyz[weighted_random_indices]
        v3_xyz = v3_xyz[weighted_random_indices]

        u = np.random.rand(num_samples,1)
        v = np.random.rand(num_samples,1)
        logging.info(f"random")
        is_a_problem = u+v>1
        u[is_a_problem] = 1 - u[is_a_problem]
        v[is_a_problem] = 1 - v[is_a_problem]
        w = 1 - (u+v)

        vs = (v1_xyz * u) + (v2_xyz * v) + (w*v3_xyz)
        logging.info(f"After vs1")
        vs = vs.astype(np.float64)
        logging.info(f"After vs2")
        ns = []
        cs = np.full_like(vs, np.array([1.0, 0.0, 0.0]))
        ns = np.asarray(ns)
        cs = np.asarray(cs)
        logging.info(f"End")
        np.cuda.Stream.null.synchronize()
        return vs.get(), ns.get(), cs.get()
