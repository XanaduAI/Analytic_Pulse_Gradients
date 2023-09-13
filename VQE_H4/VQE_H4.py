"""
Running partial VQE on H4. Partial VQE in the sense that we take a fixed Ansatz but replace part of it with a pulse gate.
This is then optimized using the analytic gradient (=generator method) or stochastic parameter shift rules.
Note that the SPS optimization is very slow and may take up to 4 hours per random seed (16 random seeds in this script).
Since each optimization is relatively low dimensional, we can utilize multiprocessing to run them embarrassingly in parallel.
This python feature is relatively new and may not work with your system. The script was run on a aws EC2 c5.2xlarge instances 
with 8 CPUs and 16GB of RAM.
"""
import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

from copy import copy

from util import run_opt, get_pulse_gates, _timeit
import concurrent.futures
import multiprocessing as mp

# Molecular Hamiltonian from quantum datasets https://pennylane.ai/datasets/qchem/h4-molecule
bond_distance = 1.0
dataset = qml.data.load("qchem", molname="H4", basis="STO-3G", bondlength=bond_distance)[0]
E_exact = dataset.fci_energy
E_vqe = dataset.vqe_energy
target_full = dataset.vqe_gates[-1]
wires = target_full.wires

# Static VQE Ansatz with singles and doubles
U = jnp.eye(2**len(dataset.hamiltonian.wires), dtype=complex)
for op in dataset.vqe_gates[:-1]:
    U = qml.matrix(op, wire_order=dataset.hamiltonian.wires) @ U
for op in dataset.vqe_gates[-1].decomposition()[:-7]:
    U = qml.matrix(op, wire_order=dataset.hamiltonian.wires) @ U

max_workers = 8 # set to the number of CPU cores available on your system
seeds = tuple(range(16))
n_epochs = 300

hyper_params = [(seeds[i], 0.05, n_epochs) for i in range(len(seeds))]
atol = 1e-15          # accuracy of ODE integration
tbins = 10            # number of time bins per pulse

T_single = 20.        # gate time for single qubit drive (on resonance)
T_CR = 100.           # gate time for two qubit drive (cross resonance)

# values taken from https://arxiv.org/pdf/1905.05670.pdf
qubit_freq = np.array([6.509, 5.963])

H_single, H_single0, H_single1, H_CR0, H_CR1 = get_pulse_gates(wires, T_single=T_single, T_CR=T_CR, qubit_freq=qubit_freq)

n_params = 5          # number of pulse gates in Ansatz

neg_mask = jnp.concatenate([-jnp.ones(tbins), jnp.ones(tbins)]) # negative amplitudes, intact phases

def qnode0(params):
    # prepare hf state, see dataset.hf_state
    for i in range(4):
        qml.PauliX(i)
    
    # fixed VQE Ansatz minus the replaced gates
    qml.QubitUnitary(U, wires = dataset.hamiltonian.wires)

    # Echoed cross resonance Ansatz, sandwiched by single qubit pulses
    qml.evolve(H_single)((params[0], params[1]), t=T_single, atol=atol)

    qml.evolve(H_CR1)((params[2],), t=T_CR, atol=atol)
    qml.PauliX(wires[1])
    qml.evolve(H_CR1)((neg_mask*params[2],), t=T_CR, atol=atol)
    qml.PauliX(wires[1])

    qml.evolve(H_single)((params[3], params[4]), t=T_single, atol=atol)
    return qml.expval(dataset.hamiltonian)

dev_jax = qml.device("default.qubit.jax", wires=dataset.hamiltonian.wires)
qnode_jax = jax.jit(qml.QNode(qnode0, dev_jax, interface="jax"))

### Run with backprop
print("Running with backprop")
value_and_grad_jax = jax.jit(jax.value_and_grad(qnode_jax))

dt_jit, dt, ddt = _timeit(value_and_grad_jax, jnp.zeros((n_params, tbins * 2)))
print(f"jit time {dt_jit}")
print(f"estimated time for optimization: ({dt*n_epochs/60} +/- {ddt*n_epochs/60}) min.")

def _wrap_run_job(hyper_params):
    seed, lr, n_epochs = hyper_params
    key = jax.random.PRNGKey(seed)
    theta0 = jax.random.normal(key, shape=(n_params, tbins * 2))
    thetaf, energy, gradients = run_opt(value_and_grad_jax, theta0, lr=lr, verbose=1, n_epochs=n_epochs, E_exact=E_exact)
    np.savez(f"data/VQE2_partial_H4_JAX-{seed}-lr-{lr}-tbins-{tbins}_n-epochs-{n_epochs}", theta=thetaf, energy=energy, gradients=gradients)

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('fork')) as executor:
    exec_map = executor.map(_wrap_run_job, hyper_params)
    tuple(circuit for circuit in exec_map)


### Run with SPS
print("running with sps and 20 split times")

# SPS not compatible with qml.Hamiltonian objects yet
# Replace qml.Hamiltonian >> qml.Sum object created by qml.dot
H_obj = qml.dot(dataset.hamiltonian.coeffs, dataset.hamiltonian.ops)

def qnode0(params):
    # prepare hf state, see dataset.hf_state
    for i in range(4):
        qml.PauliX(i)
    
    # fixed VQE Ansatz minus the replaced gates
    qml.QubitUnitary(U, wires = dataset.hamiltonian.wires)

    # Echoed cross resonance Ansatz, sandwiched by single qubit pulses
    qml.evolve(H_single)((params[0], params[1]), t=T_single, atol=atol)

    qml.evolve(H_CR1)((params[2],), t=T_CR, atol=atol)
    qml.PauliX(wires[1])
    qml.evolve(H_CR1)((neg_mask*params[2],), t=T_CR, atol=atol)
    qml.PauliX(wires[1])

    qml.evolve(H_single)((params[3], params[4]), t=T_single, atol=atol)
    return qml.expval(H_obj)

num_split_times = 20
qnode_sps = qml.QNode(qnode0, dev_jax, interface="jax", diff_method=qml.gradients.stoch_pulse_grad, use_broadcasting=True, num_split_times=num_split_times)
value_and_grad_sps = jax.value_and_grad(qnode_sps)
_ = value_and_grad_sps(jnp.zeros((n_params, tbins * 2))) # run once, not sure why but otherwise multiprocessing gets stuck, probably something about jax cache

# If for some reason multiprocessing does not work on your machine, just replace the following line with a for loop over hyper_params
def _wrap_run_job(hyper_params):
    seed, lr, n_epochs = hyper_params
    key = jax.random.PRNGKey(seed)
    theta0 = jax.random.normal(key, shape=(n_params, tbins * 2))
    thetaf, energy, gradients = run_opt(value_and_grad_sps, theta0, lr=lr, verbose=1, n_epochs=n_epochs, E_exact=E_exact)
    np.savez(f"data/VQE2_partial_H4_SPS-{seed}-lr-{lr}-tbins-{tbins}_n-epochs-{n_epochs}-num-splits-{num_split_times}", theta=thetaf, energy=energy, gradients=gradients)

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('fork')) as executor:
    exec_map = executor.map(_wrap_run_job, hyper_params)
    tuple(circuit for circuit in exec_map)

print("running with sps and 8 split times")

num_split_times = 8
qnode_sps = qml.QNode(qnode0, dev_jax, interface="jax", diff_method=qml.gradients.stoch_pulse_grad, use_broadcasting=True, num_split_times=num_split_times)
value_and_grad_sps = jax.value_and_grad(qnode_sps)
_ = value_and_grad_sps(jnp.zeros((n_params, tbins * 2)))

# If for some reason multiprocessing does not work on your machine, just replace the following line with a for loop over hyper_params
def _wrap_run_job(hyper_params):
    seed, lr, n_epochs = hyper_params
    key = jax.random.PRNGKey(seed)
    theta0 = 1.*jax.random.normal(key, shape=(n_params, tbins * 2))
    thetaf, energy, gradients = run_opt(value_and_grad_sps, theta0, lr=lr, verbose=1, n_epochs=n_epochs, E_exact=E_exact)
    np.savez(f"data/VQE2_partial_H4_SPS-{seed}-lr-{lr}-tbins-{tbins}_n-epochs-{n_epochs}-num-splits-{num_split_times}", theta=thetaf, energy=energy, gradients=gradients)

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('fork')) as executor:
    exec_map = executor.map(_wrap_run_job, hyper_params)
    tuple(circuit for circuit in exec_map)