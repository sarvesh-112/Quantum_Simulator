from flask import Flask, render_template, request, jsonify
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/inspiration')
def inspiration():
    return render_template('inspiration.html')

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.json
    mode = data.get("mode")
    shots = int(data.get("shots", 1024))
    backend = Aer.get_backend("qasm_simulator")
    result_dict = {}

    if mode == "Grover's Algorithm":
        # Grover's Algorithm for 2 qubits: Search for |11>
        qc = QuantumCircuit(2, 2)
        qc.h([0, 1])  # Superposition

        # Oracle for |11>: Z on both and CZ
        qc.cz(0, 1)

        # Diffusion operator
        qc.h([0, 1])
        qc.x([0, 1])
        qc.h(1)
        qc.cx(0, 1)
        qc.h(1)
        qc.x([0, 1])
        qc.h([0, 1])

        qc.measure([0, 1], [0, 1])

    elif mode == "Bell State":
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

    elif mode == "GHZ State":
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

    elif mode == "Shor's Algorithm (15)":
        qc = QuantumCircuit(4, 4)
        qc.h(range(4))
        qc.barrier()
        qc.measure(range(4), range(4))
        result_dict["circuit"] = fig_to_base64(qc.draw(output="mpl", style="clifford", fold=20))
        return jsonify(result_dict)

    elif mode == "Quantum Fourier Transform (QFT)":
        n = int(data.get("num_qubits", 2))
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.x(i)
        qft_circuit = QFT(n, do_swaps=True).decompose()
        qc = qc.compose(qft_circuit, qubits=range(n))
        qc.measure(range(n), range(n))

    elif mode == "Custom Qubits + Gates + Bloch Sphere":
        num_qubits = int(data.get("num_qubits", 2))
        initial_states = data.get("initial_states", ["|0>"] * num_qubits)
        gates = data.get("gates", ["None"] * num_qubits)
        cx_controls = list(map(int, data.get("cx_controls", [0] * num_qubits)))

        qc = QuantumCircuit(num_qubits, num_qubits)

        for i, state in enumerate(initial_states):
            if state == "|1>":
                qc.x(i)

        for i, gate in enumerate(gates):
            if gate == "None" or gate == "CX":
                continue
            getattr(qc, gate.lower())(i) if gate in ["X", "Y", "Z", "H", "S", "T"] else \
                qc.rx(np.pi / 2, i)

        for i, gate in enumerate(gates):
            if gate == "CX":
                control = cx_controls[i]
                if control != i:
                    qc.cx(control, i)

        circuit_img = fig_to_base64(qc.draw(output="mpl", style="clifford", fold=30))
        state = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        bloch_img = fig_to_base64(plot_bloch_multivector(state))

        amps = state.data
        state_labels = [f"|{format(i, f'0{num_qubits}b')}>" for i in range(2 ** num_qubits)]
        result_table = [{"State": label, "Amplitude": f"{amp.real:.3f} + {amp.imag:.3f}i", "Probability": f"{abs(amp) ** 2 * 100:.2f}%"} for label, amp in zip(state_labels, amps)]

        qc.measure(range(num_qubits), range(num_qubits))
        result = backend.run(transpile(qc), shots=shots).result()
        counts = result.get_counts()
        hist_img = fig_to_base64(plot_histogram(counts, bar_labels=True))

        result_dict.update({
            "circuit": circuit_img,
            "bloch": bloch_img,
            "histogram": hist_img,
            "table": result_table,
            "statevector": result_table
        })

        return jsonify(result_dict)

    else:
        return jsonify({"error": "Invalid simulation mode."})

    # For Bell, GHZ, QFT, Grover modes:
    result = backend.run(transpile(qc), shots=shots).result()
    counts = result.get_counts()

    circuit_img = fig_to_base64(qc.draw(output="mpl", style="clifford", fold=20))
    hist_img = fig_to_base64(plot_histogram(counts, bar_labels=True))

    df = pd.DataFrame.from_dict(counts, orient="index", columns=["Counts"])
    df.index.name = "State"
    df.reset_index(inplace=True)
    df["Probability"] = df["Counts"] / shots
    table = df[["State", "Probability"]].to_dict(orient="records")

    result_dict.update({
        "circuit": circuit_img,
        "histogram": hist_img,
        "bloch": None,
        "table": table,
        "statevector": []
    })

    return jsonify(result_dict)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
