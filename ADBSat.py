import os
import time
import subprocess
import shlex
import shutil
import sys
import numpy as np


def wind_projected_reference_area_from_obj(obj_file: str, flow_dir: np.ndarray, scale_to_m: float = 1.0) -> float:
    """
    Return the wind-projected reference area for an ADBSat OBJ geometry.

    The OBJ is loaded through ADBSat's own ``obj_fileTri2patch`` and
    ``surface_normals`` helpers, then evaluated as
    ``0.5 * sum(|n_i . flow_dir| A_i)``.
    """
    calc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADBSat-PyVersion", "calc")
    if calc_dir not in sys.path:
        sys.path.insert(0, calc_dir)
    from obj_fileTri2patch import obj_fileTri2patch
    from surfaceNormals import surface_normals

    _, _, x_data, y_data, z_data, _ = obj_fileTri2patch(obj_file)
    x_data = np.asarray(x_data, dtype=float) * float(scale_to_m)
    y_data = np.asarray(y_data, dtype=float) * float(scale_to_m)
    z_data = np.asarray(z_data, dtype=float) * float(scale_to_m)
    normals, areas, _ = surface_normals(x_data, y_data, z_data)

    flow = np.asarray(flow_dir, dtype=float).reshape(3)
    flow = flow / (np.linalg.norm(flow) + 1.0e-16)
    projected = 0.5 * float(np.sum(np.abs(np.asarray(areas, dtype=float) * (normals.T @ flow))))
    return float(max(projected, 1.0e-12))


class ADBSatSimulator:
    def __init__(self, method, simulation_script='python {base_dir}/simulate.py', base_dir='ADBSat-PyVersion',
                 job_template="job_adbsat.sh"):
        self.method = method
        self.simulation_script = simulation_script
        self.base_dir = os.path.abspath(base_dir)
        self.job_template = job_template

    def _resolved_simulation_script(self):
        repo_root = os.path.dirname(self.base_dir)
        return str(self.simulation_script).format(base_dir=self.base_dir, repo_root=repo_root)

    def _repo_root(self):
        return os.path.dirname(self.base_dir)

    def queue_simulation_job(self, altitude, AoS, input_file):
        """
        Erstellt und submitet einen SLURM-Job für die Simulation mit `simulate.py`,
        wartet auf den Abschluss und gibt den Jobpfad zurück.
        """
        job_subdir = os.path.join(self.base_dir, f"MFMC_Jobs_{self.method}")
        os.makedirs(job_subdir, exist_ok=True)
        job_script_path = os.path.join(job_subdir, f"job_{self.method}.sh")

        # SLURM-Skript erstellen
        stdout_path = os.path.join(job_subdir, f"ADB_Sim_{self.method}-%j.out")
        stderr_path = os.path.join(job_subdir, f"ADB_Sim_{self.method}-%j.err")
        command = shlex.split(self._resolved_simulation_script()) + [
            str(altitude),
            str(AoS),
            input_file,
        ]
        if command and command[0] in {"python", "python3"}:
            command[0] = sys.executable
        command_line = shlex.join(command)
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.method}_sim",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=128",
            "#SBATCH --time=24:00:00",
            "#SBATCH --mail-type=NONE",
            "#SBATCH --partition=cpu",
            "",
            "set -euo pipefail",
            "cd $SLURM_SUBMIT_DIR",
            "",
            f"cd {job_subdir}",
            "",
            "rm -f all_results.txt",
            command_line,
            "",
            "echo \"Simulation abgeschlossen.\""
        ]

        with open(job_script_path, "w") as f:
            f.write("\n".join(script_lines))
        os.chmod(job_script_path, 0o755)

        if shutil.which("sbatch") is None:
            result = subprocess.run(
                command,
                cwd=job_subdir,
                capture_output=True,
                text=True,
                check=True,
            )
            with open(os.path.join(job_subdir, f"ADB_Sim_{self.method}-local.out"), "w", encoding="utf-8") as f:
                f.write(result.stdout)
            with open(os.path.join(job_subdir, f"ADB_Sim_{self.method}-local.err"), "w", encoding="utf-8") as f:
                f.write(result.stderr)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            return "local", job_subdir

        # Job über sbatch einreichen
        try:
            result = subprocess.run(
                ["sbatch", job_script_path],
                cwd=self._repo_root(),
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            print(f"Simulation job submitted with ID: {job_id}")
            return job_id, job_subdir
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error submitting job: {e}")

    def _wait_for_job_completion(self, job_id, poll_interval=30):
        """
        Wartet darauf, dass ein SLURM-Job abgeschlossen wird.
        """
        if str(job_id) == "local":
            return
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
            # New format: gsi_model idx C_D C_L C_Mx C_My C_Mz cpu_time_ms
            if len(parts) >= 8:
                cd = parts[2]
                cpu_time = parts[7]
            else:
                cd = parts[2]
                cpu_time = parts[3]
            cd_value = float(cd)
            if not np.isfinite(cd_value):
                raise ValueError(
                    f"Non-finite ADBSat C_D for method={self.method}, idx={idx} in {result_file}. "
                    f"Raw line: {line.strip()}"
                )

            idx_array.append(int(float(idx)))
            Fd_values.append(cd_value)
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

        with open(result_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return (
                {k: np.asarray([], dtype=float) for k in values},
                np.asarray([], dtype=float),
                np.asarray([], dtype=int),
            )

        header_tokens = lines[0].strip().split()
        header_map = {name: pos for pos, name in enumerate(header_tokens)}
        lines = lines[1:]  # Skip header

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
                "C_L2": float("nan"),
                "C_Y": float("nan"),
                "C_Y2": float("nan"),
                "C_Mx": float("nan"),
                "C_My": float("nan"),
                "C_Mz": float("nan"),
            }

            if len(parts) >= 8:
                # Parse by column names when present to remain robust across output variants.
                def _col(name, fallback_idx=None, default=float("nan")):
                    idx = header_map.get(name, fallback_idx)
                    if idx is None or idx >= len(parts):
                        return default
                    try:
                        return float(parts[idx])
                    except Exception:
                        return default

                qoi_map["C_D"] = _col("C_D", fallback_idx=2)
                qoi_map["C_L"] = _col("C_L", fallback_idx=3)
                qoi_map["C_Y"] = _col("C_Y", default=float("nan"))
                qoi_map["C_Mx"] = _col("C_Mx", fallback_idx=4)
                qoi_map["C_My"] = _col("C_My", fallback_idx=5)
                qoi_map["C_Mz"] = _col("C_Mz", fallback_idx=6)
                cpu_time = _col("cpu_time_ms", fallback_idx=7, default=float("nan"))
            else:
                qoi_map["C_D"] = float(parts[2])
                cpu_time = float(parts[3])
            qoi_map["C_D2"] = qoi_map["C_D"] * qoi_map["C_D"] if np.isfinite(qoi_map["C_D"]) else float("nan")
            qoi_map["C_L2"] = qoi_map["C_L"] * qoi_map["C_L"] if np.isfinite(qoi_map["C_L"]) else float("nan")
            # If legacy output has no C_Y, default to 0.0 to keep Y-channel QoIs usable.
            if not np.isfinite(qoi_map["C_Y"]):
                qoi_map["C_Y"] = 0.0
            qoi_map["C_Y2"] = qoi_map["C_Y"] * qoi_map["C_Y"] if np.isfinite(qoi_map["C_Y"]) else float("nan")

            bad_qois = [
                q for q in requested_qois
                if q in qoi_map and not np.isfinite(float(qoi_map[q]))
            ]
            if bad_qois:
                raise ValueError(
                    f"Non-finite ADBSat QoI(s) {bad_qois} for method={self.method}, idx={idx} "
                    f"in {result_file}. Raw line: {line.strip()}"
                )

            idx_array.append(int(float(idx)))
            cpu_times.append(cpu_time / 3600000.0)
            for q in requested_qois:
                values.setdefault(q, []).append(float(qoi_map.get(q, float("nan"))))

        return (
            {k: np.asarray(v, dtype=float) for k, v in values.items()},
            np.asarray(cpu_times, dtype=float),
            np.asarray(idx_array, dtype=int),
        )
