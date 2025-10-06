# ===================================================================
#  Code Sample – Philippe B. Baron
#  Description: Computes 2-body and 3-body distribution functions
#               relative to an "inserted" solute using JAX for 
#               GPU acceleration. Specifically, this code quantifies
#               the local solvent structure around a solute particle
#               using 2 and 3 body correlation functions, which is useful
#               in statistical mechanical theories of solvation, and 
#               exploration of approaches to enhance solute chemical 
#               potential calculations. JAX is used to accelerate the 
#               computation of these correlation functions using GPU 
#               hardware, with batching to handle the memory vs.
#               performance tradeoff.    
#  Syntax to Run Example: 
#               python compute_g.py -p ${PWD} -n 216 -nb 100 -r 0.9 -s 3.542 -sol 1. -solv 2.
# ===================================================================

import jax
import jax.numpy as jnp
import argparse
import matplotlib.pyplot as plt
import subprocess

xyz_slice = slice(2, 5)         # slice to extract x,y,z coordinates trajectory .npz file created from a LAMMPS "dump.coord" file
query_mem = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
    encoding='utf-8',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=True
)
free_mem = int(query_mem.stdout.strip().split('\n')[0])     # obtain available GPU memory

def load_data(path):
    """Loads coordinates and box lengths from a .npz trajectory file generated from a LAMMPS dump."""
    traj_path = path + "/traj.npz"
    arr = jnp.load(traj_path)
    coords = arr["arr_0"]; boxes = arr["arr_1"]
    return coords, boxes

@jax.jit
def pbc(dr, L):
    """Apply minimum image convention to properly handle periodic box boundaries."""
    return dr - jnp.round(dr / L) * L

@jax.jit
def rdf(coords, box, I, J, bins):
    """
    Compute the radial distribution function (RDF) g(r) between two sets of particles.

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_particles, n_properties)
        Atomic coordinates and properties for a single frame. Coordinates are found at 
        the "xyz_slice" columns.
    box : jnp.ndarray
        Simulation box boundaries, can generalize NVT -> NPT.
    I : jnp.ndarray
        Indices of the first particle group (solute atoms in our case).
    J : jnp.ndarray
        Indices of the second particle group (solvent atoms in our case).
    bins : jnp.ndarray
        Bin edges for pairwise distance histogram.

    Returns
    -------
    rho : float
        Particle number density.
    bin_centers : jnp.ndarray
        Midpoints of the RDF bins.
    rdf : jnp.ndarray
        Radial distribution function values g(r).
    """

    L = box[0][1] - box[0][0]                           # Cubic box length
    Np = len(J)                                         # Number of target (solvent) particles
    rho = Np / (L ** 3)                                 # Number density
    dR = bins[1] - bins[0]                              # Bin width
    bin_centers = bins[:-1] + dR / 2                    # Midpoints of bins
    rdf = jnp.zeros_like(bins)[:-1]                     

    dr = coords[J, xyz_slice] - coords[I, xyz_slice]    # Compute vector pairwise distances
    dr_pbc = pbc(dr, L)                                 # Apply periodic boundary conditions

    Rs = jnp.linalg.norm(dr_pbc, axis=1)                # Compute scalar pairwise distances
    Rs = jnp.where(Rs > L / 2, 0.0, Rs)                 # Apply L/2 cutoff
    counts, _ = jnp.histogram(Rs, bins=bins)

    # Normalize RDF by uniform background density
    rdf = counts / (rho * 4 * jnp.pi * dR * bin_centers ** 2)
    return rho, bin_centers, rdf

@jax.jit
def g3(coords, solute, solvent, I, bins, L, n_particles, norm_arr):
    """
    Compute the three-body correlation function g^(3)(r_1, r_2, θ).

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_particles, n_properties)
        Atomic coordinates and properties for a single frame. Coordinates are found at 
        the "xyz_slice" columns.
    solute : int
        Index of solute atom.
    solvent : jnp.ndarray
        Index of reference solvent atom.
    I : jnp.ndarray
        Indices of remaining solvent atoms.
    bins : jnp.ndarray
        Binning edges for distances and angles.
    L : float
        Cubic simulation box length.
    n_particles : int
        Total number of particles in the system.
    norm_arr : jnp.ndarray
        Precomputed histogram bin volume array for normalization.

    Returns
    -------
    bin_edges : tuple of jnp.ndarray
        Edges of bins in r1, r2, and θ.
    g3 : jnp.ndarray
        Solute-solvent-solvent three-body correlation function.
    """

    bins_r = bins.copy()
    bins_th = bins.copy() * (jnp.pi / L)  # Obtain angle bins as [0, L] -> [0, π]

    # Reference solvent - solute pairwise distances
    ci = coords[solvent, xyz_slice] - coords[solute, xyz_slice]
    ci = pbc(ci, L)
    r = jnp.linalg.norm(ci) * jnp.ones_like(I)

    # Solvent - solute pairwise distances
    cj = coords[I, xyz_slice] - coords[solute, xyz_slice]
    cj = pbc(cj, L)
    r_ = jnp.linalg.norm(cj, axis=1)

    # Get distances between solvent atoms and the reference solvent atom
    dr = ci - cj
    dr_pbc = pbc(dr, L)
    dr_norm = jnp.linalg.norm(dr_pbc, axis=1)

    # Compute angle θ between r, r_
    theta = jnp.arccos((dr_norm ** 2 - r ** 2 - r_ ** 2) / (-2 * r * r_))
    
    # Combine distances and angles for 3D histogramming
    data = jnp.vstack((r, r_, theta)).T
    counts, bin_edges = jnp.histogramdd(data, bins=(bins_r, bins_r, bins_th))

    # Account for solvent reference atom indistinguishability and histogram bin volume
    g3 = (n_particles - 1) * counts / norm_arr
    return bin_edges, g3

rdf_solv = jax.vmap(rdf, (None, None, 0, None, None))                       # Vectorize rdf() over solvent atoms
rdf_solv_traj = jax.vmap(rdf_solv, (0, 0, None, None, None))                # Vectorize rdf_solv() over trajectory frames
g3_traj = jax.vmap(g3, (0, None, None, None, None, None, None, None))       # Vectorize g3() over trajectory frames

def compute_RDF(coords, boxes, inds_solute, inds_solvent, bins):
    """
    Compute the radial distribution functions (RDFs).

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_particles, n_properties)
        Atomic coordinates and properties for a single frame. Coordinates are found at 
        the "xyz_slice" columns.
    boxes : jnp.ndarray
        Array containing simulation box information for each frame.
    inds_solute : jnp.ndarray
        Index of solute atom.
    inds_solvent : jnp.ndarray
        Indices of solvent atoms.
    bins : jnp.ndarray
        Bin edges for RDF computation.

    Returns
    -------
    R : jnp.ndarray
        Radial distances corresponding to the centers of the RDF bins.
    RDF_solute : jnp.ndarray
        Solute-solvent radial distribution function averaged over frames.
    RDF_solvent : jnp.ndarray
        Solvent-solvent radial distribution function averaged over frames.
    """
    
    # Select one solvent atom as reference for solvent-solvent RDF calculation
    # This avoids NaNs in the vectorized output from divisions by zero
    chosen_ind_solvent = jnp.array([inds_solvent[0]])
    # Remaining solvent atoms for computing RDF relative to the chosen reference
    inds_solvent_reduced = inds_solvent[1:]

    # Compute solute-solvent RDF using vectorized JAX functions
    _, R, RDF_solute = rdf_solv_traj(coords, boxes, inds_solute, inds_solvent, bins)
    # Compute solvent-solvent RDF relative to the chosen reference solvent atom
    _, _, RDF_solvent = rdf_solv_traj(coords, boxes, chosen_ind_solvent, inds_solvent_reduced, bins)

    # All r-bins are identical across frames and solvent atoms, so extract from first computation
    R = R[0,0,:]

    # Average RDFs over frames
    RDF_solute = jnp.mean(RDF_solute, axis=(0,1))
    RDF_solute = RDF_solute.at[0].set(0.0)  # Ensure r=0 bin is zero
    
    RDF_solvent = jnp.mean(RDF_solvent, axis=(0,1))
    RDF_solvent = RDF_solvent.at[0].set(0.0)  # Ensure r=0 bin is zero

    return R, RDF_solute, RDF_solvent


def compute_g3(coords, n, L, inds_solvent, bins, batch_size):
    """
    Compute the three-body correlation function g3.

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_particles, n_properties)
        Atomic coordinates and properties for a single frame. Coordinates are found at 
        the "xyz_slice" columns.
    n : int
        Total number of particles in the system.
    L : float
        Cubic simulation box length (assumed NVT).
    inds_solvent : jnp.ndarray
        Indices of solvent atoms.
    bins : jnp.ndarray
        1D bin edges for g3 computation (along r dimension).
    batch_size : int
        Number of frames to process per batch to avoid exceeding GPU memory.

    Returns
    -------
    r_centers : jnp.ndarray
        Centers of radial bins used for g3 calculation.
    G2_integral : jnp.ndarray
        Integral of g3 over angular and r' bins, representing an effective pair solute-solvent RDF.
    G3 : jnp.ndarray
        Full three-body correlation function array of shape (nbins-1, nbins-1, nbins-1).
    """
    rho = n / (L ** 3)  # Particle number density

    # Compute bin centers for radial and angular bins
    r_centers = bins[:-1] + (bins[1] - bins[0]) / 2
    th_centers = r_centers.copy() * (jnp.pi / L)
    # Compute differential elements for histogram normalization
    dR = r_centers[1] - r_centers[0]
    dth = th_centers[1] - th_centers[0]

    # Create 3D meshgrid for r1, r2, and theta
    r1_grid, r2_grid, theta_grid = jnp.meshgrid(r_centers, r_centers, th_centers, indexing="ij")

    # Histogtam bin volume array for g3 based on spherical coordinates, used to normalize g3
    norm_arr = 8 * jnp.pi ** 2 * rho ** 2 * r1_grid ** 2 * r2_grid ** 2 * jnp.sin(theta_grid) * dR ** 2 * dth

    n_batches = int(len(coords) / batch_size)
    n_solv = len(inds_solvent)
    nbins = len(bins)
    G3 = jnp.zeros((nbins-1, nbins-1, nbins-1))  # Initialize accumulator for g3

    # Loop over each solvent atom to compute g3 with respect to all others
    for idx_solv in inds_solvent:

        # Single out a solvent atom to avoids NaNs in the vectorized output from divisions by zero
        inds_solute = jnp.where(coords[0,:,1] == 1.)[0]
        inds_solvent = jnp.delete(jnp.where(coords[0,:,1] == 2.)[0], idx_solv - 1)
        chosen_solvent_ind = jnp.array([idx_solv])

        # Process frames in batches to avoid GPU memory issues
        for idx_batch in range(n_batches):
            c_batch = coords[idx_batch * batch_size : (idx_batch + 1) * batch_size]
            _, g3 = g3_traj(c_batch, inds_solute, chosen_solvent_ind, inds_solvent, bins, L, n, norm_arr)
            # Average over batch and normalize by number of batches and solvent atoms
            G3 += jnp.mean(g3, axis=0) / (n_batches * n_solv)
    
    # Convert g3 into a solute-solvent RDF by integrating over theta and r' bins
    G3_i = 2 * jnp.pi * r2_grid ** 2 * jnp.sin(theta_grid) * G3 * rho / (n - 2)
    G2_integral = jnp.trapezoid(jnp.trapezoid(G3_i, th_centers, axis=2), r_centers, axis=1)

    return r_centers, G2_integral, G3

def main(path, n, nbins, rho_star, sigma, solute, solvent, plot_flag=True):
    """
    Controls computation of g(2) and g(3) correlation functions from a trajectory relative to 
    an "inserted" solute atom [Currently, this code handles a single inserted solute into a dense,
    homogeneous liquid]. This "particle insertion" construction is useful for chemical potential
    calculations for a dense liquid. Computed correlation functions are saved in "g.npz".

    Parameters
    ----------
    path : str
        Directory path containing trajectory ('traj.npz') with coordinates and box info from simulation.
    n : int
        Total number of particles in the system.
    nbins : int
        Number of bins for RDF and three-body histograms.
    rho_star : float
        Reduced number density in LJ units.
    sigma : float
        Lennard-Jones length scale in Angstroms.
    solute : float
        Identifier for solute particle.
    solvent : float
        Identifier for solvent particles.
    """

    L = ((n * sigma ** 3) / rho_star) ** (1 / 3)                            # Cubic box length determined from reduced density and sigma
    cutoff = L/2                                                            # Set interatomic cutoff to half the box length
    rho = n / (L ** 3)                                                      # Particle number density                     

    # Load simulation coordinates and box lengths (not needed for NVT, allows extension to NPT)
    coords, boxes = load_data(path)

    # Determine batch size based on available GPU memory
    jax_mem_overhead = 10                                                   # Empirical overhead estimate for JAX vectorization for g3 (heuristic, may vary by system)
    vec_mem = jax_mem_overhead * (8 / (1024 ** 2)) * (10 * n + nbins ** 3)
    batch_size = int(free_mem / vec_mem)
    
    bins = jnp.linspace(0, cutoff, nbins)                                   # Set bin edges for RDF calculation
    g3_bins = jnp.linspace(0, L, nbins)                                     # Set 1D bin edges for g3 calculation

    inds_solute = jnp.where(coords[0,:,1] == solute)[0]                     # Index of "inserted" solute atom
    inds_solvent = jnp.where(coords[0,:,1] == solvent)[0]                   # Indices of solvent atoms

    # Compute 2-body and 3-body correlation functions relative to solute
    R, RDF_solute, RDF_solvent = compute_RDF(coords, boxes, inds_solute, inds_solvent, bins)
    r_centers, G2_integral, G3 = compute_g3(coords, n, L, inds_solvent, g3_bins, batch_size)

    # Plot results [optional] to verify correctness
    # All RDFs should decay to 1 and G2_integral should closely match RDF_solute
    if plot_flag:
        plt.figure()
        plt.plot(R, RDF_solute, label=r"solute-solvent $g^{(2)}(r)$")
        plt.plot(R, RDF_solvent, label=r"solvent-solvent $g^{(2)}(r)$")
        plt.plot(r_centers, G2_integral, label=r"solute-solvent $g^{(2)}(r)$ from $g^{(3)}(r,r',\theta)$")
        plt.axhline(y = 1, color = 'k', linestyle = '--')
        plt.xlabel("r (Å)")
        plt.ylabel(r"$g^{(2)}(r)$")
        plt.legend()
        plt.show()
        plt.savefig("rdf.png")
    
    # Save correlation functions for future use
    jnp.savez("g.npz", R, RDF_solute, RDF_solvent, G3)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", type=str, help="base path to state 0, 1 simulations")
    argParser.add_argument("-n", "--n_particles", type=int, help="total number of atoms")
    argParser.add_argument("-nb", "--n_bins", type=int, help="number of bins to use in g2, g3 computation")
    argParser.add_argument("-r", "--rho", type=float, help="system density in LJ reduced units")
    argParser.add_argument("-s", "--sigma", type=float, help="sigma parameter in Angstroms")
    argParser.add_argument("-sol", "--solute", type=float, help="solute particle identifier")
    argParser.add_argument("-solv", "--solvent", type=float, help="solvent particle identifier")
    args = argParser.parse_args()
    main(args.path, args.n_particles, args.n_bins, args.rho, args.sigma, args.solute, args.solvent)