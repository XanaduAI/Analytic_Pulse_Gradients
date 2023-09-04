import optax
import numpy as np
import jax

import pennylane as qml
import jax.numpy as jnp
import jax

import time

from copy import copy

from datetime import datetime

T_single = 50.
T_CR = 200.

def drive_field(T, wdrive):
    def wrapped(p, t):
        """ callable phi and omega drive with 4 slots """
        # The first len(p)-1 values of the trainable params p characterize the pwc function
        amp = qml.pulse.pwc(T)(p[:len(p)//2], t)
        phi = qml.pulse.pwc(T)(p[len(p)//2:], t)
        wd = wdrive # + qml.pulse.pwc(T)(p[-tomegas:], t)
        #amp = max_amp*normalize(amp)
        return amp * jnp.sin(wd * t + phi)

    return wrapped

def get_pulse_gates(wires, T_single=T_single, T_CR=T_CR, qubit_freq=[6.509, 5.963]):

    # https://arxiv.org/pdf/1905.05670.pdf
    qubit_freq = 2 * np.pi * np.array(qubit_freq)

    H0 = qml.dot(-0.5*qubit_freq, [qml.PauliZ(i) for i in wires])
    H0 += 2*np.pi*0.0123 * (qml.PauliX(wires[0]) @ qml.PauliX(wires[1]) + qml.PauliY(wires[0]) @ qml.PauliY(wires[1]))

    H_single = copy(H0)
    for q,i in enumerate(wires):
        H_single += drive_field(T_single, qubit_freq[q]) * qml.PauliY(i)

    H_single0 = copy(H0)
    H_single0 += drive_field(T_single, qubit_freq[0]) * qml.PauliY(wires[0])

    H_single1 = copy(H0)
    H_single1 += drive_field(T_single, qubit_freq[1]) * qml.PauliY(wires[1])

    H_CR0 = copy(H0)
    H_CR1 = copy(H0)
    H_CR0 += drive_field(T_CR, qubit_freq[1]) * qml.PauliY(wires[0])
    H_CR1 += drive_field(T_CR, qubit_freq[0]) * qml.PauliY(wires[1])
    return H_single, H_single0, H_single1, H_CR0, H_CR1


def run_opt_jit(value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, E_exact=0., verbose=True):

    # The following block creates a constant schedule of the learning rate
    # that increases from 0.1 to 0.5 after 10 epochs
    # schedule0 = optax.constant_schedule(1e-1)
    # schedule1 = optax.constant_schedule(5e-1)
    # schedule = optax.join_schedules([schedule0, schedule1], [10])
    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def step(theta, opt_state):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)
        return val, theta, grad_circuit, opt_state

    # ## Compile the evaluation and gradient function and report compilation time
    # time0 = datetime.now()
    # _ = value_and_grad(theta)
    # time1 = datetime.now()
    # print(f"grad and val compilation time: {time1 - time0}")
    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        val, theta, grad_circuit, opt_state = step(theta, opt_state)
        # val, grad_circuit = value_and_grad(theta)
        # updates, opt_state = optimizer.update(grad_circuit, opt_state)
        # theta = optax.apply_updates(theta, updates)

        energy[n] = val
        gradients.append(
            grad_circuit
        )
        thetas.append(
            theta
        )
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val - E_exact}; min loss: {np.min(energy) - E_exact}; after {t1 - t0}")
    
    return thetas, energy, gradients

def run_opt(value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, E_exact=0., verbose=True):

    # The following block creates a constant schedule of the learning rate
    # that increases from 0.1 to 0.5 after 10 epochs
    # schedule0 = optax.constant_schedule(1e-1)
    # schedule1 = optax.constant_schedule(5e-1)
    # schedule = optax.join_schedules([schedule0, schedule1], [10])
    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def partial_step(grad_circuit, opt_state, theta):
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta


    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        # val, theta, grad_circuit, opt_state = step(theta, opt_state)
        val, grad_circuit = value_and_grad(theta)
        opt_state, theta = partial_step(grad_circuit, opt_state, theta)

        energy[n] = val
        gradients.append(
            grad_circuit
        )
        thetas.append(
            theta
        )
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val - E_exact}; min loss: {np.min(energy) - E_exact}; after {t1 - t0}")
    
    return thetas, energy, gradients

def _timeit(callable, *args, reps=10):
    #callable.clear_cache()

    jittime0 = time.process_time()
    _ = jax.block_until_ready(callable(*args))
    dt_jit = time.process_time() - jittime0 

    dts = []
    for k in range(reps):
        t0 = time.process_time()
        _ = jax.block_until_ready(callable(*args))
        dt = time.process_time() - t0

        dts.append(dt)
    
    return dt_jit, np.mean(dts), np.std(dts)