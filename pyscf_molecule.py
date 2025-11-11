import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, fci, lo


def to_np_array(a):
    if isinstance(a, (np.ndarray, jnp.ndarray)):
        return np.array(a)
    elif isinstance(a, (tuple, list, dict)):
        return jax.tree_util.tree_map(to_np_array, a)

def variables_from_mol(mol):
    coords = to_np_array(mol.coordinates)  # (N_n, d)

    n_dets = 1
    n_up, n_dn = mol.n_per_spin

    ind_orb_up = np.arange(n_up).reshape(n_dets, -1)
    ind_orb_dn = np.arange(n_dn).reshape(n_dets, -1)
    ind_orb = [ind_orb_up, ind_orb_dn]

    mo_coeff = to_np_array(mol.mf_mo_coefficients)  # (n_orbitals, n_orbitals)

    # added from PauliNet
    mo_coeff = jnp.asarray(mo_coeff)
    ao_overlap = mol.ao_overlap
    mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))

    mo_coeff_spin = [mo_coeff[:,:n_up], mo_coeff[:,:n_dn]]  # they want it different per spin
    atomic_orbitals = mol.atomic_orbitals()

    return coords, atomic_orbitals, mo_coeff_spin, ind_orb



class Molecule:
    """This class is to access the molecule and SCF calculations from pyscf."""
    def __init__(
            self,
            geometry,
            ecp=None,
            run_fci=True,
            verbose=True,
            charge=0,
            spin=0,
            basis='sto-3g',
            unit="Bohr"
    ):
        self.mol = gto.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, ecp=ecp)
        self.mol.cart = True
        self.unit = unit
        assert unit.lower() == "bohr", "that's the correct one!"

        self._mf = None
        self._fci = None
        if verbose:
            print("Running restricted UHF (!)... (might want RHF)")
            print(self.mf.kernel())
        if verbose:
            print("UHF done.")
        self.ao_overlap = self.mf.mol.intor('int1e_ovlp_cart')
        if run_fci:
            if verbose:
                print("Running fci...")
                self.fci
            if verbose:
                print("fci done.")

    def info(self):
        print(self.mol.dump_input())

    @property
    def mf(self):
        "Hartree Fock energy"
        if self._mf is None:
            self._mf = scf.HF(self.mol).run()
        return self._mf

    @property
    def fci(self):
        if self._fci is None:
            self._fci = fci.FCI(self.mf).kernel()
        return self._fci

    def mo_boys(self, verbose=0):
        mo_boys = lo.Boys(self.mol).kernel(self.mf.mo_coeff, verbose=verbose)
        return mo_boys

    @property
    def n_basis(self):
        return [self.mol.bas_nprim(i) for i in range(self.mol.nbas)]
        # return [self.mol.bas_atom(i) for i in range(self.mol.nbas)]

    @property
    def coordinates(self):
        return self.mol.atom_coords(unit=self.unit)

    @property
    def n_orbitals(self):
        return self.mf.mo_coeff.shape[0]
        # return self.mol.nbas # somehow, this one gave wrong results sometimes

    def atomic_orbitals(self):
        "Parameters etc. for the atomic orbitals (HF)."
        mol = self.mol
        assert mol.cart
        orbitals = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = mol.bas_exp(i)
            coeff_sets = mol.bas_ctr_coeff(i).T  # !!!
            # print(l, coeff_sets, _get_cartesian_angulars(l))
            for coeffs in coeff_sets:  # !!! this part I was missing
                # shells.append((i_atom, (l, coeffs, zetas)))
                # for lxyz in _get_cartesian_angulars(l):
                ao = {
                    "ind_nuc": i_atom,
                    "zetas": zetas,  # or alphas?
                    "weights": coeffs,
                    "ang_mom": l
                }
                orbitals.append(ao)
        # assert len(orbitals) == self.n_orbitals, f"got orbitals = {len(orbitals)} vs n_orbitals = {self.n_orbitals}"
        return orbitals

    @property
    def basis_set(self):
        return self.mol.basis

    @property
    def nuclear_charges(self):
        return self.mol.atom_charges()

    @property
    def n_electrons(self):
        return self.mol.tot_electrons()

    @property
    def n_per_spin(self):
        return self.mol.nelec

    @property
    def mf_mo_coefficients(self):
        "Coefficients from HF to combine atomic orbitals."
        # warnings.warn("taking mean-field mo_coeff")
        return self.mf.mo_coeff

    @property
    def EHF(self):
        return sum(self.mf.scf_summary.values())

    @property
    def EFCI(self):
        return self.fci[0]
    
    @property
    def ecp(self):
        return self.mol.ecp

    def __repr__(self):
        return f"Molecule(\n  {self.mol._atom},\n  basis={self.basis_set},\n  n_orbitals={self.n_orbitals},\n  n_electrons={self.n_per_spin}\n)"
