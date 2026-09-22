import os
import shlex
import sys
import time
import subprocess
import numpy as np


PRESSURE_QOIS = ("P_D", "P_L", "P_Y", "P_Mx", "P_My", "P_Mz")


def _pressure_result_row(header_tokens, parts):
    """Parse one ADBSat row and return area-normalized loads in Pa."""
    columns = {name: pos for pos, name in enumerate(header_tokens)}

    def _column(name, default=float("nan")):
        pos = columns.get(name)
        if pos is None or pos >= len(parts):
            return default
        try:
            return float(parts[pos])
        except (TypeError, ValueError):
            return default

    pressure_values = {name: _column(name) for name in PRESSURE_QOIS}
    q_inf = _column("q_inf")
    if not np.isfinite(pressure_values["P_D"]):
        if not np.isfinite(q_inf) or q_inf <= 0.0:
            raise ValueError(
                "ADBSat result row contains aerodynamic coefficients but no positive q_inf; "
                "pressure in Pa cannot be reconstructed. Re-run the simulation with the pressure output format."
            )
        coefficient_names = ("C_D", "C_L", "C_Y", "C_Mx", "C_My", "C_Mz")
        pressure_values = {
            pressure_name: _column(coefficient_name) * q_inf
            for pressure_name, coefficient_name in zip(PRESSURE_QOIS, coefficient_names)
        }

    pressure_values["P_D2"] = pressure_values["P_D"] ** 2
    pressure_values["P_L2"] = pressure_values["P_L"] ** 2
    pressure_values["P_Y2"] = pressure_values["P_Y"] ** 2
    coefficient_values = {}
    for pressure_name, coefficient_name in zip(
        PRESSURE_QOIS,
        ("C_D", "C_L", "C_Y", "C_Mx", "C_My", "C_Mz"),
    ):
        coefficient_values[coefficient_name] = (
            pressure_values[pressure_name] / q_inf
            if np.isfinite(q_inf) and q_inf > 0.0
            else float("nan")
        )
    coefficient_values["C_D2"] = coefficient_values["C_D"] ** 2
    coefficient_values["C_L2"] = coefficient_values["C_L"] ** 2
    coefficient_values["C_Y2"] = coefficient_values["C_Y"] ** 2
    pressure_values.update(coefficient_values)
    cpu_time_ms = _column("cpu_time_ms")
    return pressure_values, cpu_time_ms


class ADBSatSimulator:
    def __init__(self, method, simulation_script=None, base_dir='ADBSat-PyVersion',
                 job_template="job_adbsat.sh", cpus_per_task=36):
        self.method = method
        self.base_dir = os.path.abspath(base_dir)
        if simulation_script is None:
            simulation_script = (
                f"{shlex.quote(os.path.abspath(sys.executable))} "
                f"{shlex.quote(os.path.join(self.base_dir, 'simulate.py'))}"
            )
        self.simulation_script = simulation_script
        self.job_template = job_template
        self.cpus_per_task = int(cpus_per_task)

    def queue_simulation_job(self, altitude, AoS, input_file):
        """
        Erstellt und submitet einen SLURM-Job für die Simulation mit `simulate.py`,
        wartet auf den Abschluss und gibt den Jobpfad zurück.
        """
        job_subdir = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}")
        os.makedirs(job_subdir, exist_ok=True)
        job_script_path = os.path.join(job_subdir, f"job_{self.method}.sh")

        # SLURM-Skript erstellen
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.method}_sim",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            "#SBATCH --time=24:00:00",
            "#SBATCH --partition=prandtl",
            f"#SBATCH --output=ADB_Sim_{self.method}-%j.out",
            f"#SBATCH --error=ADB_Sim_{self.method}-%j.err",
            "",
            "echo \"Arbeitsverzeichnis: $SLURM_SUBMIT_DIR\"",
            "cd $SLURM_SUBMIT_DIR",
            "",
            "module load gcc",
            "module load openmpi",
            "module load hdf5",
            "",
            f"cd {job_subdir}",
            "",
            f"{self.simulation_script} {altitude} {AoS} {input_file}",
            "",
            "echo \"Simulation abgeschlossen.\""
        ]

        with open(job_script_path, "w") as f:
            f.write("\n".join(script_lines))
        os.chmod(job_script_path, 0o755)

        # Job über sbatch einreichen
        try:
            result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"Simulation job submitted with ID: {job_id}")
            return job_id, job_subdir
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error submitting job: {e}")

    def _wait_for_job_completion(self, job_id, poll_interval=30):
        """
        Wartet darauf, dass ein SLURM-Job abgeschlossen wird.
        """
        print(f"Warte auf SLURM Job {job_id}...")
        while True:
            try:
                result = subprocess.run(["squeue", "--job", job_id], capture_output=True, text=True)
                if job_id not in result.stdout:
                    print(f"Job {job_id} abgeschlossen.")
                    break
            except subprocess.CalledProcessError:
                break
            time.sleep(poll_interval)

    def analyze_simulation_results(self, indices):
        """
        Liest `all_results.txt` und gibt P_D [Pa] und CPU-Zeiten zurück.
        """
        result_file = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}", "all_results.txt")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Results file {result_file} not found!")

        Fd_values = []
        cpu_times = []
        idx_array = []

        indices_set = set(map(str, indices))

        with open(result_file, "r") as f:
            all_lines = f.readlines()
        header_tokens = all_lines[0].strip().split() if all_lines else []
        lines = all_lines[1:]

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            gsi_model = parts[0]
            idx = parts[1]
            if gsi_model != self.method or idx not in indices_set:
                continue

            qoi_map, cpu_time = _pressure_result_row(header_tokens, parts)

            idx_array.append(int(float(idx)))
            Fd_values.append(float(qoi_map["P_D"]))
            cpu_times.append(float(cpu_time) / 3600000.0)  # ms → h

        return np.array(Fd_values), np.array(cpu_times), np.array(idx_array)

    def analyze_simulation_results_qois(self, indices, requested_qois=None):
        """
        Read all_results.txt and return requested pressure QoIs in Pa.
        Returns:
            values_by_qoi: dict[str, np.ndarray]
            cpu_times_h: np.ndarray
            idx_array: np.ndarray
        """
        if requested_qois is None:
            requested_qois = ["P_D"]

        result_file = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}", "all_results.txt")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Results file {result_file} not found!")

        idx_array = []
        cpu_times = []
        values = {q: [] for q in requested_qois}
        indices_set = set(map(str, indices))

        with open(result_file, "r") as f:
            all_lines = f.readlines()
        header_tokens = all_lines[0].strip().split() if all_lines else []
        lines = all_lines[1:]

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            gsi_model = parts[0]
            idx = parts[1]
            if gsi_model != self.method or idx not in indices_set:
                continue

            qoi_map, cpu_time = _pressure_result_row(header_tokens, parts)

            idx_array.append(int(float(idx)))
            cpu_times.append(cpu_time / 3600000.0)
            for q in requested_qois:
                values.setdefault(q, []).append(float(qoi_map.get(q, float("nan"))))

        return (
            {k: np.asarray(v, dtype=float) for k, v in values.items()},
            np.asarray(cpu_times, dtype=float),
            np.asarray(idx_array, dtype=int),
        )
