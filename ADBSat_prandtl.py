import os
import shlex
import sys
import time
import subprocess
import numpy as np

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
            "module load gcc/12.3.0",
            "module load openmpi/4.1.5",
            "module load hdf5/1.12.2",
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
        Liest `all_results.txt` und gibt Fd-Werte und CPU-Zeiten zurück.
        """
        result_file = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}", "all_results.txt")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Results file {result_file} not found!")

        Fd_values = []
        cpu_times = []
        idx_array = []

        indices_set = set(map(str, indices))

        with open(result_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            gsi_model = parts[0]
            idx = parts[1]
            if gsi_model != self.method or idx not in indices_set:
                continue

            # Backward compatibility:
            # Old format: gsi_model idx Cd cpu_time_ms
            # New formats:
            #   gsi_model idx C_D C_L C_Mx C_My C_Mz cpu_time_ms
            #   gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz cpu_time_ms
            if len(parts) >= 9:
                cd = parts[2]
                cpu_time = parts[8]
            elif len(parts) >= 8:
                cd = parts[2]
                cpu_time = parts[7]
            else:
                cd = parts[2]
                cpu_time = parts[3]

            idx_array.append(int(float(idx)))
            Fd_values.append(float(cd))
            cpu_times.append(float(cpu_time) / 3600000.0)  # ms → h

        return np.array(Fd_values), np.array(cpu_times), np.array(idx_array)

    def analyze_simulation_results_qois(self, indices, requested_qois=None):
        """
        Read all_results.txt and return requested QoIs with costs and indices.
        Returns:
            values_by_qoi: dict[str, np.ndarray]
            cpu_times_h: np.ndarray
            idx_array: np.ndarray
        """
        if requested_qois is None:
            requested_qois = ["C_D"]

        result_file = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}", "all_results.txt")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Results file {result_file} not found!")

        idx_array = []
        cpu_times = []
        values = {q: [] for q in requested_qois}
        indices_set = set(map(str, indices))

        with open(result_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            gsi_model = parts[0]
            idx = parts[1]
            if gsi_model != self.method or idx not in indices_set:
                continue

            qoi_map = {
                "C_D": float("nan"),
                "C_D2": float("nan"),
                "C_L": float("nan"),
                "C_Y": float("nan"),
                "C_Mx": float("nan"),
                "C_My": float("nan"),
                "C_Mz": float("nan"),
            }

            if len(parts) >= 9:
                qoi_map["C_D"] = float(parts[2])
                qoi_map["C_L"] = float(parts[3])
                qoi_map["C_Y"] = float(parts[4])
                qoi_map["C_Mx"] = float(parts[5])
                qoi_map["C_My"] = float(parts[6])
                qoi_map["C_Mz"] = float(parts[7])
                cpu_time = float(parts[8])
            elif len(parts) >= 8:
                qoi_map["C_D"] = float(parts[2])
                qoi_map["C_L"] = float(parts[3])
                qoi_map["C_Mx"] = float(parts[4])
                qoi_map["C_My"] = float(parts[5])
                qoi_map["C_Mz"] = float(parts[6])
                cpu_time = float(parts[7])
            else:
                qoi_map["C_D"] = float(parts[2])
                cpu_time = float(parts[3])
            qoi_map["C_D2"] = qoi_map["C_D"] * qoi_map["C_D"] if np.isfinite(qoi_map["C_D"]) else float("nan")

            idx_array.append(int(float(idx)))
            cpu_times.append(cpu_time / 3600000.0)
            for q in requested_qois:
                values.setdefault(q, []).append(float(qoi_map.get(q, float("nan"))))

        return (
            {k: np.asarray(v, dtype=float) for k, v in values.items()},
            np.asarray(cpu_times, dtype=float),
            np.asarray(idx_array, dtype=int),
        )
