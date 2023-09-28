import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax
import optax

jax.config.update("jax_enable_x64", True)

bond_distance = 1.78
data = qml.data.load("qchem", molname="HeH+", basis="STO-3G", bondlength=bond_distance)[0]
H_obj = data.tapered_hamiltonian
H_obj = qml.dot(np.array(H_obj.coeffs), H_obj.ops)
E_exact = data.fci_energy

# values and parametrization from https://arxiv.org/pdf/2210.15812.pdf
# all in units of 10^9
qubit_freq = jnp.pi * 2 * np.array([5.23, 5.01])
eps = np.array([32.9, 31.5]) #10^9
max_amp = jnp.array([0.955, 0.987]) # much larger than in ctrl-vqe paper
connections = [(0, 1)]
coupling = 0.0123
wires = [0, 1]
n_wires = len(wires)
dt = 0.22
timespan = dt * 720 # 360

def normalize(x):
    """Differentiable normalization to +/- 1 outputs (shifted sigmoid)"""
    return (1 - jnp.exp(-x))/(1 + jnp.exp(-x))

legendres = jnp.array([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    0.5*jnp.array([0, 0, 3, 0, -1]),
    0.5*jnp.array([0, 5, 0, -3, 0]),
    1/8*jnp.array([35, 0, -30, 0, 3])
])
leg_func = jax.jit(jax.vmap(jnp.polyval, [0, None]))
dLeg = len(legendres)

def amp(timespan, omega, max_amp):
    def wrapped(p, t):
        pr, pi = p[:dLeg], p[dLeg:]
        par = pr + 1j * pi
        leg_vals = leg_func(legendres, 2*t/timespan - 1)
        z = par @ leg_vals
        res = normalize(z) * jnp.angle(z)
        res = max_amp * jnp.real(jnp.exp(1j*omega*t) * res) # eq. (27)
        return res
    return wrapped

H_D = qml.dot(0.5*eps, [qml.Identity(i) - qml.PauliZ(i) for i in wires])
H_D += coupling/2 * (qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)) # implicit factor 2 due to redundancy in formula

fs = [amp(timespan, qubit_freq[i], max_amp[i]) for i in range(n_wires)]
ops = [qml.PauliX(i) for i in wires]

H_C = qml.dot(fs, ops)
H = H_D + H_C

atol=1e-8

dev = qml.device("default.qubit.jax", wires=n_wires)

def circuit(params):
    qml.evolve(H, atol=atol)(params, t=timespan)
    return qml.expval(H_obj)

def f(params, tau):
    return [fs[i](params[i], tau) for i in range(len(fs))]

num_split_times = 20

cost_ps8 = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.stoch_pulse_grad, 
    num_split_times=8, 
    use_broadcasting=True
)
cost_ps20 = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.stoch_pulse_grad, 
    num_split_times=20, 
    use_broadcasting=True
)
cost_gen1 = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.pulse_odegen, 
    atol=1., 
    use_broadcasting=True
)
cost_jax = qml.QNode(circuit, dev, interface="jax")

value_and_grad_ps8 = jax.value_and_grad(cost_ps8)
value_and_grad_ps20 = jax.value_and_grad(cost_ps20)
value_and_grad_gen1 = jax.value_and_grad(cost_gen1)
value_and_grad_jax = jax.jit(jax.value_and_grad(cost_jax))

def run_opt(value_and_grad, theta, n_epochs=100):

    optimizer = optax.adam(learning_rate=0.02)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []

    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        energy[n] = val
        gradients.append(
            grad_circuit
        )
    t1 = datetime.now()
    print(f"Final energy discrepancy: {val-E_exact} after {t1 - t0}")
    
    return theta, energy, gradients

for seed in np.arange(100):
    print(f"seed {seed+1} / 100")
    key = jax.random.PRNGKey(seed)
    theta0 = jax.random.normal(key, shape=(n_wires, 2*dLeg))

    thetaf, energy, gradients = run_opt(value_and_grad_jax, theta0)
    np.savez(f"data/VQE_Maryland__HeH_jax_{seed}", theta=thetaf, energy=energy, gradients=gradients)

    thetaf, energy, gradients = run_opt(value_and_grad_gen1, theta0)
    np.savez(f"data/VQE__HeH+-1.78_atol-1_{seed}", theta=thetaf, energy=energy, gradients=gradients)

    thetaf, energy, gradients = run_opt(value_and_grad_ps8, theta0)
    np.savez(f"data/VQE_Maryland__HeH+-1.78_ps-split-8_{seed}", theta=thetaf, energy=energy, gradients=gradients)

    thetaf, energy, gradients = run_opt(value_and_grad_ps20, theta0)
    np.savez(f"data/VQE_Maryland__HeH+-1.78_ps-split-20_{seed}", theta=thetaf, energy=energy, gradients=gradients)
