import jax
import jax.numpy as jnp
import numpy as np

geometries = {
        'LiH' : [('Li', (0,0,0)),
                ('H', (0,0,3.015))],
        'dimer' : [
            ('C',(15.2590006687, 16.8121532299, 15.9738065311)),
            ('O',(9.4486299386, 13.5060171431, 14.45823873)),
            ('H',(15.4105888182, 17.5070205934, 17.9029011803)),
            ('H',(15.2544898927, 18.4025938643, 14.6708461317)),
            ('H',(16.8591544947, 15.5899710576, 15.5578683925)),
            ('H',(13.5117713588, 15.749029294,  15.7636104197)),
            ('H',(7.68753004, 13.1203202896, 14.2934357266)),
            ('H',(10.167173679, 13.0333702175, 12.8652242888))
            ],
        'dimer_' : [
            ('C', (29.448623, 17.819358, 18.344269)),
            ('H', (30.915398, 19.144469, 18.909845)),
            ('H', (27.708179, 18.253679, 19.348981)),
            ('H', (29.128138, 17.975003, 16.319352)),
            ('H', (30.042775, 15.904281, 18.798897226)),
            ('O', (9.448630, 12.577577, 12.030276)),
            ('H', (10.973795, 12.307386, 11.093119)),
            ('H', (8.430457, 11.125273, 11.667596))
            ],
        'CH4' : [
            ('C', (29.448623, 17.819358, 18.344269)),
            ('H', (30.915398, 19.144469, 18.909845)),
            ('H', (27.708179, 18.253679, 19.348981)),
            ('H', (29.128138, 17.975003, 16.319352)),
            ('H', (30.042775, 15.904281, 18.798897226))
            ],
        'H2O' : [
            ('O', (9.4486299386, 12.5775758681, 12.0302753173)),
            ('H', (10.9737937682, 12.3073847361, 11.093118626)),
            ('H', (8.4304569148,  11.1252723137, 11.6675953263))
        ],
        'Ga' : [('Ga', (0, 0, 0))],
        'Kr' : [('Kr', (0, 0, 0))],
        'Sc' : [('Sc', (0, 0, 0))],
        'Li' : [('Li', (0, 0, 0))],
        'Be' : [('Be', (0, 0, 0))],
        'B' : [('B', (0, 0, 0))],
        'C' : [('C', (0, 0, 0))],
        'H' : [('H', (0, 0, 0))],
        'H2' : [
            ('H', (0, 0, 0)),
            ('H', (0, 0, 1.400287))
            ],

        'ScO' : [
            ('Sc', (0, 0, 0)),
            ('O', (0, 0, 3.2088))
            ],
        'BeH2' : [
            ('Be', (0, 0, 0)),
            ('H', (0, 0, 2.5303433)),
            ('H', (0, 0, -2.5303433))
            ],
        
    }

def make_geometry(name_mol):
    return geometries[name_mol]


  
def get_el_ion_distance_matrix(r_el, R_ion):
    """
    Computes distance vectors and their norm between inputs
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = r_el[..., None, :] - R_ion[..., None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist

def get_full_distance_matrix(r_el):
    """
    Computes distance vecttors between inputs
    Args:
        r_el: shape [n_el x 3]
    Returns:
    """
    diff = jnp.expand_dims(r_el, -2) - jnp.expand_dims(r_el, -3)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist

def dists_from_diffs_matrix(r_el_diff):
    n_el = r_el_diff.shape[-2]
    diff_padded = r_el_diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return dist

def get_distance_matrix(r_el): #  stable!
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)
    Args:
        r_el: [batch_dims x n_electrons x 3]
    Returns:
        tuple: differences [batch_dims x n_el x n_el x 3], distances [batch_dims x n_el x n_el]
    """
    diff = r_el[..., :, None, :] - r_el[..., None, :, :]
    dist = dists_from_diffs_matrix(diff)
    return diff, dist

def make_ecp(geometry):
    ecp = {}
    unique_atoms = set(atom for atom, _ in geometry)
    for atom in unique_atoms:
        ecp.update({atom:'ccecp'})
    return ecp