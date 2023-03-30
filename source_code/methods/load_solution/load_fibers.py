from dolfin import *
import pandas as pd
import numpy as np

def load_fibers(load_fibers_dir):

    f = HDF5File(mpi_comm_world(),load_fibers_dir + 'solution.hdf5','r')


    mesh = Mesh()

    f.read(mesh,"/mesh",False)

    VQuadelem = VectorElement("Quadrature", mesh.ufl_cell(), degree=2, quad_scheme="default")
    VQuadelem._quad_scheme = 'default'

    fiberFS = FunctionSpace(mesh, VQuadelem)
    f0 = Function(fiberFS)
    s0 = Function(fiberFS)
    n0 = Function(fiberFS)

    f.read(f0,"f0")
    f.read(s0,"s0")
    f.read(n0,"n0")


    fiber_loaded = {
        "mesh": mesh,
        "f0": f0,
        "s0": s0,
        "n0": n0,
    }

    return fiber_loaded
