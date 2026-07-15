import os
import time
import glob
import zipfile
import shutil
import subprocess
import shlex
import json
import uuid
import numpy as np
import meshio
import scipy.io as sio
import pyvista as pv


GEOMETRY_MESH_ALIASES = {
    "CUBE": "Cube_mesh.h5",
    "SOAR": "SOAR_mesh.h5",
    "GOCE": "GOCE_mesh.h5",
    "CHAMP": "CHAMP_mesh.h5",
}

PROJECT_NAME_ALIASES = {
    "CUBE": "Cube",
    "SOAR": "SOAR",
    "GOCE": "GOCE",
    "CHAMP": "CHAMP",
}


def _canonical_geometry_key(value) -> str:
    if value is None:
        return "CUBE"
    return str(value).strip().upper()


def _boundary3_source_name(geometry_id=None, geometry_mesh=None) -> str:
    if _canonical_geometry_key(geometry_id) == "CUBE":
        return "CUBE"
    if geometry_mesh:
        mesh_name = os.path.basename(str(geometry_mesh)).strip().lower()
        if mesh_name == "cube_mesh.h5":
            return "CUBE"
    return "OBJ"


def _project_name_from_geometry(geometry_id=None, geometry_mesh=None) -> str:
    for candidate in (geometry_id, geometry_mesh):
        if candidate is None:
            continue
        token = os.path.basename(str(candidate)).strip()
        if not token:
            continue
        lower = token.lower()
        if lower.endswith("_mesh.h5"):
            token = token[:-8]
        elif lower.endswith(".h5"):
            token = token[:-3]
        if token:
            return PROJECT_NAME_ALIASES.get(token.upper(), token)
    return "Cube"


def _mesh_filename_from_geometry(geometry_id=None, geometry_mesh=None, geometry_mesh_map=None) -> str:
    if geometry_mesh:
        mesh_name = os.path.basename(str(geometry_mesh)).strip()
        if mesh_name:
            return mesh_name

    mesh_map = geometry_mesh_map if isinstance(geometry_mesh_map, dict) else GEOMETRY_MESH_ALIASES
    mesh_name = mesh_map.get(_canonical_geometry_key(geometry_id), "Cube_mesh.h5")
    return os.path.basename(str(mesh_name))


def _rewrite_job_ini_geometry(ini_path: str, mesh_file: str, project_name: str, source_name: str) -> None:
    with open(ini_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated = []
    for line in lines:
        if line.lstrip().startswith("MeshFile"):
            updated.append(f"MeshFile = {mesh_file}  ! (relative) path to meshfile\n")
        elif line.lstrip().startswith("ProjectName"):
            updated.append(f"ProjectName     = {project_name}    ! Name of the current simulation\n")
        elif line.lstrip().startswith("Part-Boundary3-SourceName"):
            updated.append(f"Part-Boundary3-SourceName  = {source_name}\n")
        else:
            updated.append(line)

    with open(ini_path, "w", encoding="utf-8") as f:
        f.writelines(updated)


def _patch_piclas_collision_mode(
    ini_path: str,
    collision_mode=None,
    calc_quality_factors=None,
) -> None:
    if collision_mode is None and calc_quality_factors is None:
        return

    with open(ini_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated = []
    found_dsmc = False
    found_collis = False
    found_quality = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("UseDSMC"):
            updated.append("UseDSMC                          = T                                          ! Flag for using DSMC in Calculation\n")
            found_dsmc = True
        elif stripped.startswith("Particles-DSMC-CalcQualityFactors") and calc_quality_factors is not None:
            flag = "T" if bool(calc_quality_factors) else "F"
            updated.append(
                f"Particles-DSMC-CalcQualityFactors = {flag}       "
                "! Enables / disables the calculation and output of quality factors\n"
            )
            found_quality = True
        elif stripped.startswith("Particles-DSMC-CollisMode") and collision_mode is not None:
            updated.append(
                f"Particles-DSMC-CollisMode        = {int(collision_mode)}                                          "
                "! Define mode of collision handling in DSMC\n"
            )
            found_collis = True
        else:
            updated.append(line)

    if not found_dsmc:
        updated.append("UseDSMC                          = T                                          ! Flag for using DSMC in Calculation\n")
    if calc_quality_factors is not None and not found_quality:
        flag = "T" if bool(calc_quality_factors) else "F"
        updated.append(
            f"Particles-DSMC-CalcQualityFactors = {flag}       "
            "! Enables / disables the calculation and output of quality factors\n"
        )
    if collision_mode is not None and not found_collis:
        updated.append(
            f"Particles-DSMC-CollisMode        = {int(collision_mode)}                                          "
            "! Define mode of collision handling in DSMC\n"
        )

    with open(ini_path, "w", encoding="utf-8") as f:
        f.writelines(updated)


def _resolve_piclas_collision_controls(piclas_mode, collision_mode, calc_quality_factors):
    mode = "" if piclas_mode is None else str(piclas_mode).strip().lower()
    if mode in {"tpmc", "collisionless", "free_molecular", "free-molecular"}:
        if collision_mode is None:
            collision_mode = 0
        if calc_quality_factors is None:
            calc_quality_factors = False
    elif mode in {"dsmc", "collisional"}:
        if collision_mode is None:
            collision_mode = 2
        if calc_quality_factors is None:
            calc_quality_factors = True

    if collision_mode is None:
        return None, calc_quality_factors
    return int(collision_mode), calc_quality_factors


def _rotation_matrix_z(aos_deg: float) -> np.ndarray:
    aos_rad = np.radians(aos_deg)
    return np.array(
        [
            [np.cos(aos_rad), -np.sin(aos_rad), 0.0],
            [np.sin(aos_rad), np.cos(aos_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_matrix_x(aoa_deg: float) -> np.ndarray:
    aoa_rad = np.radians(aoa_deg)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(aoa_rad), -np.sin(aoa_rad)],
            [0.0, np.sin(aoa_rad), np.cos(aoa_rad)],
        ]
    )


def _expand_values(value, count: int, default: float) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return [float(default)] * count
        if arr.size >= count:
            return [float(v) for v in arr[:count]]
        pad = [float(arr[-1])] * (count - arr.size)
        return [float(v) for v in arr.tolist()] + pad
    try:
        scalar = float(value)
    except Exception:
        scalar = float(default)
    return [scalar] * count


def _expand_int_values(value, count: int, default: int) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return [int(default)] * count
        vals = []
        for item in arr[:count]:
            try:
                vals.append(int(item))
            except Exception:
                vals.append(int(default))
        if len(vals) < count:
            vals.extend([vals[-1] if vals else int(default)] * (count - len(vals)))
        return vals
    try:
        scalar = int(value)
    except Exception:
        scalar = int(default)
    return [scalar] * count


def _extract_force_per_area_cell(mesh: pv.DataSet) -> np.ndarray:
    """
    Return force-per-area as cell array with shape (n_cells, 3) when possible.
    Falls back to scalar shape (n_cells, 1) for legacy outputs.
    """
    if "Total_ForcePerArea" in mesh.cell_data:
        arr = np.asarray(mesh.cell_data["Total_ForcePerArea"])
    elif "Total_ForcePerArea" in mesh.point_data:
        cell_mesh = mesh.point_data_to_cell_data(pass_point_data=False)
        if "Total_ForcePerArea" not in cell_mesh.cell_data:
            raise KeyError("Total_ForcePerArea could not be converted from point_data to cell_data")
        arr = np.asarray(cell_mesh.cell_data["Total_ForcePerArea"])
    else:
        raise KeyError("Total_ForcePerArea not found in mesh cell_data or point_data")

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _normalize_flow_zero_direction(flow_zero_direction=None):
    if flow_zero_direction is None:
        return None
    value = flow_zero_direction
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) != 3:
            return None
        value = parts
    try:
        vec = np.asarray(value, dtype=float).reshape(-1)[:3]
    except Exception:
        return None
    if vec.size < 3:
        return None
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vec / norm


def _flow_unit_from_angles(aos_deg: float, aoa_deg: float = 0.0, flow_zero_direction=None) -> np.ndarray:
    zero_dir = _normalize_flow_zero_direction(flow_zero_direction)
    if zero_dir is not None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        horizontal_zero = zero_dir - float(np.dot(zero_dir, up)) * up
        if float(np.linalg.norm(horizontal_zero)) <= 1e-12:
            horizontal_zero = np.array([1.0, 0.0, 0.0], dtype=float)
        horizontal_zero = horizontal_zero / (np.linalg.norm(horizontal_zero) + 1e-16)
        side = np.cross(horizontal_zero, up)
        side = side / (np.linalg.norm(side) + 1e-16)
        aos_rad = np.radians(float(aos_deg))
        aoa_rad = np.radians(float(aoa_deg))
        horizontal = np.cos(aos_rad) * horizontal_zero + np.sin(aos_rad) * side
        flow = np.cos(aoa_rad) * horizontal + np.sin(aoa_rad) * up
        return flow / (np.linalg.norm(flow) + 1e-16)
    return (_rotation_matrix_z(aos_deg) @ _rotation_matrix_x(aoa_deg)) @ np.array([0.0, 1.0, 0.0])


def _force_frame_axes(aos_deg: float, aoa_deg: float = 0.0, flow_zero_direction=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flow/side/lift axes.
    Flow direction follows update_parameter_file convention with AoS + AoA support.
    """
    flow_dir = _flow_unit_from_angles(aos_deg, aoa_deg, flow_zero_direction)
    flow_dir = flow_dir / (np.linalg.norm(flow_dir) + 1e-16)

    up_ref = np.array([0.0, 0.0, 1.0])
    side_dir = np.cross(flow_dir, up_ref)
    if np.linalg.norm(side_dir) < 1e-12:
        side_dir = np.array([1.0, 0.0, 0.0])
    side_dir = side_dir / (np.linalg.norm(side_dir) + 1e-16)

    lift_dir = np.cross(side_dir, flow_dir)
    lift_dir = lift_dir / (np.linalg.norm(lift_dir) + 1e-16)
    return flow_dir, side_dir, lift_dir


def cell_areas_and_total(vtu_file: str):
    """
    Compute the area of every surface-cell in a VTU file and the total area.

    Parameters
    ----------
    vtu_file : str
        Path to the *.vtu* surface-mesh file.

    Returns
    -------
    areas : numpy.ndarray
        1-D array (length = n_cells) with the area of each cell in the
        order PyVista stores them.
    area_total : float
        Sum of all cell areas.
    """
    # --- load mesh -----------------------------------------------------------
    grid = pv.read(vtu_file)

    # --- let VTK do the heavy lifting ---------------------------------------
    #     compute_cell_sizes adds a new cell-data array called "Area"
    #     (length and volume are disabled to save time).
    grid_sz      = grid.compute_cell_sizes(length=False, area=True, volume=False)
    areas        = grid_sz.cell_data["Area"]          # vtkDataArray → NumPy-view
    area_total   = float(areas.sum())                 # convert to Python scalar

    return np.asarray(areas), area_total


def wind_projected_reference_area(vtu_file: str, flow_dir: np.ndarray, areas=None) -> float:
    """
    Return the projected reference area normal to the freestream direction.

    For a closed surface this is the silhouette area, evaluated as
    0.5 * sum(|n_i . flow_dir| A_i). This replaces the old 0.5*wetted-area
    convention and keeps the reference tied to the actual wind direction.
    """
    grid = pv.read(vtu_file)
    flow = np.asarray(flow_dir, dtype=float).reshape(3)
    flow = flow / (np.linalg.norm(flow) + 1e-16)
    if not hasattr(grid, "n_cells"):
        if areas is None:
            raise AttributeError("Surface mesh does not expose n_cells and no fallback areas were provided")
        return float(max(0.5 * np.sum(np.asarray(areas, dtype=float)), 1e-12))
    area_vector_dots = []
    for cell_idx in range(grid.n_cells):
        points = np.asarray(grid.get_cell(cell_idx).points, dtype=float)
        if points.shape[0] < 3:
            area_vector_dots.append(0.0)
            continue
        area_vector = np.zeros(3, dtype=float)
        origin = points[0]
        for point_idx in range(1, points.shape[0] - 1):
            area_vector += 0.5 * np.cross(points[point_idx] - origin, points[point_idx + 1] - origin)
        area_vector_dots.append(abs(float(np.dot(area_vector, flow))))
    projected = 0.5 * np.sum(area_vector_dots)
    return float(max(projected, 1e-12))

class Simulator:
    def run(self, altitude, AoS, db_index, random_seed, count=1, wait_for_completion=True):
        raise NotImplementedError("Simulator.run must be implemented in a subclass.")


class PiclasSimulator:
    def __init__(self,
                 update_script='python update_parameter.py',
                 update_dir='update_parameter_file',
                 piclas_dir='piclas',
                 fortran_exe='piclas',
                 mpi_procs=36,
                 ini_low='DSMC1.ini',
                 ini_high='parameter.ini',
                 output_files=["output1.vtu", "output2.vtu", "output3.vtu", "output4.vtu"],
                 job_template="job_piclas.sh",
                 geometry_mesh_map=None,
                 debug_geometry=False,
                 debug_geometry_dir="debug_geometry",
                 piclas_mode=None,
                 collision_mode=None,
                 collis_mode=None,
                 calc_quality_factors=None,
                 flow_zero_direction=None,
                 required_surface_state_files=None,
                 node_cores=64,
                 submission_group_size=10,
                 submit_sleep_s=0.0):
        self.update_script = update_script
        self.update_dir = os.path.abspath(update_dir)
        self.piclas_dir = os.path.abspath(piclas_dir)
        self.fortran_exe = fortran_exe
        self.mpi_procs = mpi_procs
        self.ini_low = ini_low
        self.ini_high = ini_high
        self.output_files = output_files
        self.job_template = job_template
        self.debug_geometry = bool(debug_geometry)
        self.debug_geometry_dir = str(debug_geometry_dir)
        self.piclas_mode = "" if piclas_mode is None else str(piclas_mode).strip().lower()
        if collision_mode is None and collis_mode is not None:
            collision_mode = collis_mode
        self.collision_mode, self.calc_quality_factors = _resolve_piclas_collision_controls(
            piclas_mode,
            collision_mode,
            calc_quality_factors,
        )
        if required_surface_state_files is None:
            required_surface_state_files = len(self.output_files)
        self.required_surface_state_files = max(1, int(required_surface_state_files))
        self.flow_zero_direction = flow_zero_direction
        self.node_cores = max(1, int(node_cores))
        self.submission_group_size = max(1, int(submission_group_size))
        self.submit_sleep_s = max(0.0, float(submit_sleep_s))
        self.geometry_mesh_map = {k: v for k, v in GEOMETRY_MESH_ALIASES.items()}
        if isinstance(geometry_mesh_map, dict):
            for k, v in geometry_mesh_map.items():
                self.geometry_mesh_map[_canonical_geometry_key(k)] = str(v)

    def _output_vtu_files(self, job_subdir):
        pattern = os.path.join(job_subdir, "output*.vtu")
        files = sorted(glob.glob(pattern))
        if files:
            return files
        legacy_files = [os.path.join(job_subdir, fname) for fname in self.output_files]
        return [path for path in legacy_files if os.path.exists(path)]

    def _resolve_mesh_source(self, geometry_id=None, geometry_mesh=None):
        if geometry_mesh:
            mesh_filename = str(geometry_mesh)
        else:
            mesh_filename = self.geometry_mesh_map.get(_canonical_geometry_key(geometry_id), "Cube_mesh.h5")

        mesh_path = mesh_filename
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(self.piclas_dir, mesh_filename)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file for geometry '{geometry_id}' not found: {mesh_path}")
        return mesh_path

    def _resolve_project_name(self, geometry_id=None, geometry_mesh=None):
        return _project_name_from_geometry(geometry_id=geometry_id, geometry_mesh=geometry_mesh)

    def _resolve_mesh_filename(self, geometry_id=None, geometry_mesh=None):
        return _mesh_filename_from_geometry(
            geometry_id=geometry_id,
            geometry_mesh=geometry_mesh,
            geometry_mesh_map=self.geometry_mesh_map,
        )

    def _job_project_name(self, job_subdir):
        ini_path = os.path.join(job_subdir, self.ini_high)
        if not os.path.exists(ini_path):
            return "Cube"
        with open(ini_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.lstrip().startswith("ProjectName"):
                    continue
                value = line.split("=", 1)[1].split("!", 1)[0].strip()
                return value or "Cube"
        return "Cube"

    def _make_job_subdir_name(self, db_index, aos, geometry_id=None):
        geo = _canonical_geometry_key(geometry_id or "cube").lower()
        try:
            aos_val = float(aos)
        except Exception:
            aos_val = 0.0
        aos_token = f"{aos_val:+06.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        unique = f"{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        return f"job_{geo}_db{int(db_index)}_aos{aos_token}_{unique}"

    def _geometry_debug_paths(self, job_subdir):
        debug_root = os.path.join(self.piclas_dir, self.debug_geometry_dir)
        os.makedirs(debug_root, exist_ok=True)
        stem = os.path.basename(job_subdir)
        return {
            "root": debug_root,
            "update_json": os.path.join(debug_root, f"{stem}_update_parameter.json"),
            "piclas_json": os.path.join(debug_root, f"{stem}_piclas_prepare.json"),
        }

    def prepare_simulation_folder(
        self,
        altitude,
        AoS,
        db_index,
        random_seed=None,
        env_payload_path=None,
        env_model=None,
        geometry_id=None,
        geometry_mesh=None,
    ):
        subdir_name = self._make_job_subdir_name(db_index=db_index, aos=AoS, geometry_id=geometry_id)
        job_subdir = os.path.join(self.piclas_dir, subdir_name)
        os.makedirs(job_subdir, exist_ok=True)
        update_cmd = shlex.split(self.update_script) + [str(altitude), str(AoS), str(db_index), str(self.ini_high)]
        if random_seed is not None:
            update_cmd.extend(["--random-seed", str(int(random_seed))])
        debug_paths = self._geometry_debug_paths(job_subdir) if self.debug_geometry else None
        if env_payload_path:
            update_cmd.extend(["--env-payload", str(env_payload_path)])
        if env_model:
            update_cmd.extend(["--env-model", str(env_model)])
        if geometry_id:
            update_cmd.extend(["--geometry-id", str(geometry_id)])
        if geometry_mesh:
            update_cmd.extend(["--geometry-mesh", str(geometry_mesh)])
        if debug_paths is not None:
            update_cmd.extend(["--debug-print", "--debug-json", str(debug_paths["update_json"])])
        subprocess.run(update_cmd, cwd=self.update_dir, check=True)
        job_ini_path = os.path.join(job_subdir, self.ini_high)
        shutil.copy(os.path.join(self.update_dir, self.ini_high), job_ini_path)
        mesh_filename = self._resolve_mesh_filename(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        project_name = self._resolve_project_name(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        source_name = _boundary3_source_name(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        _rewrite_job_ini_geometry(job_ini_path, mesh_file=mesh_filename, project_name=project_name, source_name=source_name)
        _patch_piclas_collision_mode(
            job_ini_path,
            collision_mode=self.collision_mode,
            calc_quality_factors=self.calc_quality_factors,
        )
        shutil.copy(os.path.join(self.update_dir, 'dyn_p.txt'), os.path.join(job_subdir, 'dyn_p.txt'))
        for filename in [self.ini_low, 'piclas', 'piclas2vtk']:
            shutil.copy(os.path.join(self.piclas_dir, filename), os.path.join(job_subdir, filename))
        mesh_src = self._resolve_mesh_source(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        shutil.copy(mesh_src, os.path.join(job_subdir, mesh_filename))
        if debug_paths is not None:
            debug_payload = {
                "job_subdir": job_subdir,
                "geometry_id": geometry_id,
                "geometry_mesh_argument": geometry_mesh,
                "env_payload_path": env_payload_path,
                "env_model": env_model,
                "update_cmd": update_cmd,
                "resolved_mesh_source": mesh_src,
                "resolved_mesh_filename": mesh_filename,
                "resolved_project_name": project_name,
                "resolved_boundary3_source_name": source_name,
                "job_ini_path": job_ini_path,
                "debug_update_json": debug_paths["update_json"],
            }
            with open(debug_paths["piclas_json"], "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, indent=2, sort_keys=True)
            print(
                "[PICLas DEBUG] "
                f"job_subdir={job_subdir} geometry_id={geometry_id} "
                f"mesh_filename={mesh_filename} project_name={project_name} "
                f"boundary3_source_name={source_name} update_debug_json={debug_paths['update_json']}"
            )
        return job_subdir

    def create_simulation_job_script(self, job_subdir):
        script_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=piclas_sim",
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node={self.mpi_procs}",
            "#SBATCH --time=01:00:00",
            "#SBATCH --partition=prandtl",
            "#SBATCH --output=piclas_slurm-%j.out",
            "#SBATCH --error=piclas_slurm-%j.err",
            "",
            "ulimit -l 83886080",
            "module load gcc/12.3.0",
            "module load openmpi/4.1.5",
            "module load hdf5/1.12.2",
            "",
            f"cd {job_subdir}",
            "",
            "start=$(date +%s%3N)",
            f"mpirun -np {self.mpi_procs} ./piclas parameter.ini DSMC1.ini",
            "end=$(date +%s%3N)",
            "runtime=$((end - start))",
            "echo $runtime > cpu_time.txt"
        ]

        job_script_path = os.path.join(job_subdir, self.job_template)
        with open(job_script_path, "w") as f:
            f.write("\n".join(script_lines))
        os.chmod(job_script_path, 0o755)
        return job_script_path

    def submit_simulation_job(self, job_script_path):
        result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id

    def create_simulation_group_job_script(self, job_subdirs):
        requested_tasks = max(1, int(self.mpi_procs))
        group_dir = os.path.join(self.piclas_dir, "group_jobs")
        os.makedirs(group_dir, exist_ok=True)
        group_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        job_script_path = os.path.join(group_dir, f"piclas_group_{group_id}.sh")

        script_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=piclas_group",
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node={requested_tasks}",
            "#SBATCH --time=01:00:00",
            "#SBATCH --partition=prandtl",
            "#SBATCH --output=piclas_group-%j.out",
            "#SBATCH --error=piclas_group-%j.err",
            "",
            "ulimit -l 83886080",
            "module load gcc/12.3.0",
            "module load openmpi/4.1.5",
            "module load hdf5/1.12.2",
            "",
            "run_case() {",
            "  local case_dir=\"$1\"",
            "  echo \"PICLas case ${case_dir}\"",
            "  cd \"$case_dir\" || exit 1",
            "  start=$(date +%s%3N)",
            f"  mpirun -np {self.mpi_procs} ./piclas parameter.ini DSMC1.ini",
            "  status=$?",
            "  end=$(date +%s%3N)",
            "  runtime=$((end - start))",
            "  echo $runtime > cpu_time.txt",
            "  return $status",
            "}",
            "",
        ]
        for subdir in job_subdirs:
            script_lines.append(f"run_case {shlex.quote(subdir)} || exit 1")

        with open(job_script_path, "w") as f:
            f.write("\n".join(script_lines))
        os.chmod(job_script_path, 0o755)
        return job_script_path

    def _query_job_states(self, job_ids=None):
        cmd = ["squeue", "-h", "-o", "%i %t"]
        if job_ids:
            normalized = [str(job_id) for job_id in job_ids]
            cmd.extend(["--jobs", ",".join(normalized)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            return {}

        states = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            states[parts[0]] = parts[1].strip()
        return states

    def _wait_for_job_completion(self, job_id, poll_interval=10):
        tracked_job_id = str(job_id)
        while True:
            job_states = self._query_job_states([tracked_job_id])
            state = job_states.get(tracked_job_id)
            if state is None or state == "CG":
                break
            time.sleep(poll_interval)

    def _wait_for_all_jobs_completion(self, job_ids, poll_interval=60):
        pending_job_ids = {str(job_id) for job_id in job_ids}
        while True:
            if not pending_job_ids:
                break
            job_states = self._query_job_states(sorted(pending_job_ids))
            pending_job_ids = {
                job_id
                for job_id in pending_job_ids
                if job_states.get(job_id) not in (None, "CG")
            }
            if not pending_job_ids:
                break
            print(f"Warte auf Abschluss von {len(pending_job_ids)} Jobs...")
            time.sleep(poll_interval)
        print("Alle Jobs abgeschlossen!")

    def _surface_state_glob_pattern(self, job_subdir=None, geometry_id=None, geometry_mesh=None):
        if job_subdir is not None:
            project_name = self._job_project_name(job_subdir)
        else:
            project_name = self._resolve_project_name(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        return f"{project_name}_DSMCSurfState_000.00*"

    def _visu_surface_glob_pattern(self, job_subdir=None, geometry_id=None, geometry_mesh=None):
        if job_subdir is not None:
            project_name = self._job_project_name(job_subdir)
        else:
            project_name = self._resolve_project_name(geometry_id=geometry_id, geometry_mesh=geometry_mesh)
        return f"{project_name}_visuSurf_000.00*"

    def _surface_state_files(self, job_subdir):
        pattern = os.path.join(job_subdir, self._surface_state_glob_pattern(job_subdir=job_subdir))
        return sorted(glob.glob(pattern))

    def _has_required_surface_state_outputs(self, job_subdir):
        return len(self._surface_state_files(job_subdir)) >= self.required_surface_state_files

    def _cleanup_job_outputs_for_retry(self, job_subdir):
        cleanup_patterns = [
            self._surface_state_glob_pattern(job_subdir=job_subdir),
            self._visu_surface_glob_pattern(job_subdir=job_subdir),
            "output*.vtu",
            "cpu_time.txt",
        ]
        for pattern in cleanup_patterns:
            for path in glob.glob(os.path.join(job_subdir, pattern)):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except FileNotFoundError:
                    continue

    def _wait_for_jobs_and_retry_failed_outputs(self, job_subdirs, job_ids, max_retries=2):
        pending_job_ids = list(job_ids)
        retry_count = 0

        while True:
            if pending_job_ids:
                print(f"Warte auf Abschluss aller {len(pending_job_ids)} Simulationen...")
                self._wait_for_all_jobs_completion(pending_job_ids)

            failed_subdirs = [subdir for subdir in job_subdirs if not self._has_required_surface_state_outputs(subdir)]
            if not failed_subdirs:
                return

            if retry_count >= max_retries:
                missing_outputs = ", ".join(
                    f"{os.path.basename(subdir)} "
                    f"({len(self._surface_state_files(subdir))}/{self.required_surface_state_files} H5)"
                    for subdir in failed_subdirs
                )
                raise RuntimeError(
                    "Maximale Anzahl an PICLas-Retries erreicht. Fehlende H5-Outputs für: "
                    f"{missing_outputs}"
                )

            retry_count += 1
            print(
                f"{len(failed_subdirs)} Jobs ohne vollständige H5-Outputs. "
                f"Starte Retry {retry_count}/{max_retries}."
            )

            pending_job_ids = []
            for subdir in failed_subdirs:
                self._cleanup_job_outputs_for_retry(subdir)
                job_script_path = os.path.join(subdir, self.job_template)
                if not os.path.exists(job_script_path):
                    job_script_path = self.create_simulation_job_script(subdir)
                if self.submit_sleep_s > 0.0:
                    time.sleep(self.submit_sleep_s)
                job_id = self.submit_simulation_job(job_script_path)
                print(f"Retry-Job {job_id} für {os.path.basename(subdir)} gestartet.")
                pending_job_ids.append(job_id)

    def submit_postprocessing_job(self, list_of_job_subdirs, random_seed, wait_for_completion=True):
        postproc_dir = os.path.join(self.piclas_dir, "postprocessing")
        os.makedirs(postproc_dir, exist_ok=True)
        script_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=piclas_postproc",
            "#SBATCH --nodes=1",
            f"#SBATCH --ntasks={self.node_cores}",
            f"#SBATCH --cpus-per-task=1",
            "#SBATCH --time=00:30:00",
            "#SBATCH --partition=prandtl",
            "#SBATCH --output=piclas_postproc-%j.out",
            "#SBATCH --error=piclas_postproc-%j.err",
            "",
            "module load gcc/12.3.0",
            "module load openmpi/4.1.5",
            "module load hdf5/1.12.2",
            "",
            "MAX_PARALLEL=${SLURM_NTASKS:-64}",
            "running=0",
            "postprocess_case() {",
            "  local case_dir=\"$1\"",
            "  local surface_pattern=\"$2\"",
            "  local visu_pattern=\"$3\"",
            "  echo \"Postprocessing in ${case_dir}...\"",
            "  cd \"$case_dir\" || exit 1",
            "  ./piclas2vtk $surface_pattern",
            "  rm -f output*.vtu",
            "  mapfile -t files < <(ls $visu_pattern 2>/dev/null)",
            "  if [ ${#files[@]} -lt 1 ]; then echo \"No VTU files in $PWD\" && exit 1; fi",
            "  i=1",
            "  for f in \"${files[@]}\"; do",
            "    printf -v out_name 'output%04d.vtu' \"$i\"",
            "    mv \"$f\" \"$out_name\"",
            "    i=$((i + 1))",
            "  done",
            "}",
            "",
            "# Fuehre piclas2vtk parallel ueber alle Job-Unterordner aus",
        ]
        for subdir in list_of_job_subdirs:
            surface_pattern = self._surface_state_glob_pattern(job_subdir=subdir)
            visu_pattern = self._visu_surface_glob_pattern(job_subdir=subdir)
            script_lines.extend([
                f"postprocess_case {shlex.quote(subdir)} {shlex.quote(surface_pattern)} {shlex.quote(visu_pattern)} &",
                "running=$((running + 1))",
                "if [ \"$running\" -ge \"$MAX_PARALLEL\" ]; then",
                "  wait -n || exit 1",
                "  running=$((running - 1))",
                "fi",
            ])
        script_lines.extend([
            "while [ \"$running\" -gt 0 ]; do",
            "  wait -n || exit 1",
            "  running=$((running - 1))",
            "done",
        ])
        script_content = "\n".join(script_lines)
        postproc_script = os.path.join(
            postproc_dir,
            f"postproc_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.sh",
        )
        with open(postproc_script, "w") as f:
            f.write(script_content)
        os.chmod(postproc_script, 0o755)
        result = subprocess.run(["sbatch", postproc_script], capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        if wait_for_completion:
            self._wait_for_job_completion(job_id)
        return job_id

    def collect_results(self, job_subdirs):
        """
        Liefert für jedes Unterverzeichnis
            • globalen Gesamt-C_D  (Bezug A_ref = windprojizierte Fläche)
            • CPU-Stunden
        -------------------------------------------------------------------------
        Rückgabe
        --------
        global_cd_list  : list[float]   # Gesamt-C_D (pro Job)
        cpu_hours_list  : list[float]   # Laufzeit in Stunden (pro Job)
        """
        aw_cd_list, global_cd_list, cpu_hours_list = [], [], []

        for subdir in job_subdirs:
            # ---------------------------------------------------------------------
            # Dynamischer Druck
            # ---------------------------------------------------------------------
            dyn_p = float(np.loadtxt(os.path.join(subdir, "dyn_p.txt")))

            # ---------------------------------------------------------------------
            # Zellflächen & Referenzfläche einmal aus *einem* VTU einlesen
            # (wir nehmen das erste verfügbare VTU)
            # ---------------------------------------------------------------------
            output_files = self._output_vtu_files(subdir)
            if not output_files:
                raise FileNotFoundError(f"Keine output*.vtu Dateien gefunden in {subdir}")
            area_file = output_files[0]
            areas, _A_wetted = cell_areas_and_total(area_file)
            flow_dir, _, _ = _force_frame_axes(0.0, 0.0, self.flow_zero_direction)
            A_total = wind_projected_reference_area(area_file, flow_dir, areas)
            A_ref = A_total

            # Listen für die einzelnen Zeitschritte
            aw_cds, global_cds = [], []

            result_files = output_files[1:] if len(output_files) > 1 else output_files
            for fpath in result_files:
                mesh = pv.read(fpath)

                # --------------------------------------------------------------
                # Total_ForcePerArea kann in cell_data *oder* point_data liegen.
                #   • wenn Vektor -> Betrag
                # --------------------------------------------------------------
                if "Total_ForcePerArea" in mesh.cell_data_dict:
                    # mesh.cell_data_dict[celltype] → ndarray(n_cells, 3|1)
                    cell_force = next(iter(mesh.cell_data_dict["Total_ForcePerArea"]))
                elif "Total_ForcePerArea" in mesh.point_data:
                    cell_force = mesh.point_data["Total_ForcePerArea"]
                else:
                    raise KeyError(f"Total_ForcePerArea nicht gefunden in {fpath}")

                cell_force = np.asarray(cell_force)
                if cell_force.ndim == 2:               # Vektorfeld → Norm
                    cell_force = np.linalg.norm(cell_force, axis=1)

                if cell_force.size != areas.size:
                    raise ValueError(f"Zellzahl passt nicht zu Flächen in {fpath}")

                # --------------------------------------------------------------
                # 1) Fläche-gewichtetes Mittel-C_D
                #    \bar{C_D} =  Σ (f_i * A_i) / (q * A_total_projected)
                # --------------------------------------------------------------
                aw_cd = np.sum(cell_force * areas) / (dyn_p * A_total)

                # --------------------------------------------------------------
                # 2) Gesamt-C_D   (Referenzfläche A_ref)
                #    C_D = Σ (f_i * A_i) / (q * A_ref_projected)
                # --------------------------------------------------------------
                global_cd = np.sum(cell_force * areas) / (dyn_p * A_ref)

                aw_cds.append(aw_cd)
                global_cds.append(global_cd)

            # Mittel über alle Zeitschritte dieses Jobs
            aw_cd_list.append(float(np.mean(aw_cds)))
            global_cd_list.append(float(np.mean(global_cds)))

            # -----------------------------------------------------------------                                                                                                                     
            # CPU-Zeit (ms) → h
            # -----------------------------------------------------------------
            with open(os.path.join(subdir, "cpu_time.txt")) as f:
                cpu_time_ms = float(f.read().strip())
            cpu_hours_list.append(cpu_time_ms * self.mpi_procs / 3_600_000.0)

        return global_cd_list, cpu_hours_list

    def collect_results_qois(self, job_subdirs, AoS, AoA=0.0, flow_zero_direction=None):
        """
        Collect drag, lift, side-force, and moment coefficients (if vector force data is available).
        Always returns C_D; C_L/C_Y/C_M* may be NaN for scalar-only solver output.
        """
        qoi_values = {
            "C_D": [],
            "C_D2": [],
            "C_L": [],
            "C_Y": [],
            "C_Mx": [],
            "C_My": [],
            "C_Mz": [],
        }
        cpu_hours_list = []
        aos_values = _expand_values(AoS, len(job_subdirs), 0.0)
        aoa_values = _expand_values(AoA, len(job_subdirs), 0.0)

        for idx, subdir in enumerate(job_subdirs):
            active_zero = flow_zero_direction if flow_zero_direction is not None else self.flow_zero_direction
            flow_dir, side_dir, lift_dir = _force_frame_axes(float(aos_values[idx]), float(aoa_values[idx]), active_zero)
            dyn_p = float(np.loadtxt(os.path.join(subdir, "dyn_p.txt")))
            output_files = self._output_vtu_files(subdir)
            if not output_files:
                raise FileNotFoundError(f"Keine output*.vtu Dateien gefunden in {subdir}")
            area_file = output_files[0]
            areas, _A_wetted = cell_areas_and_total(area_file)
            A_ref = wind_projected_reference_area(area_file, flow_dir, areas)
            # Preserve the established moment convention while drag/lift use
            # the wind-projected reference area.
            L_ref = float(np.sqrt(max(A_wetted, 1e-12)))

            cds, cls, cys = [], [], []
            cmx_list, cmy_list, cmz_list = [], [], []

            result_files = output_files[1:] if len(output_files) > 1 else output_files
            for fpath in result_files:
                mesh = pv.read(fpath)
                force_pa = _extract_force_per_area_cell(mesh)
                if force_pa.shape[0] != areas.size:
                    raise ValueError(f"Zellzahl passt nicht zu Flächen in {fpath}")

                # Scalar force fallback: drag-only.
                if force_pa.shape[1] == 1:
                    scalar_f = force_pa[:, 0]
                    # Keep legacy drag behavior for scalar-only solver output.
                    drag = float(abs(np.sum(scalar_f * areas) / (dyn_p * A_ref)))
                    lift = float("nan")
                    side_force = float("nan")
                    cm_vec = np.array([float("nan"), float("nan"), float("nan")])
                else:
                    # Vector force integration.
                    force_vec = force_pa[:, :3]
                    total_force = np.sum(force_vec * areas.reshape(-1, 1), axis=0)
                    c_vec = total_force / (dyn_p * A_ref)

                    # Drag is opposite freestream; lift along lift axis.
                    drag = float(abs(-np.dot(c_vec, flow_dir)))
                    lift = float(np.dot(c_vec, lift_dir))
                    side_force = float(np.dot(c_vec, side_dir))

                    centers = mesh.cell_centers().points
                    moments = np.sum(np.cross(centers, force_vec * areas.reshape(-1, 1)), axis=0)
                    cm_vec = moments / (dyn_p * A_ref * L_ref)

                cds.append(drag)
                cls.append(lift)
                cys.append(side_force)
                cmx_list.append(float(cm_vec[0]))
                cmy_list.append(float(cm_vec[1]))
                cmz_list.append(float(cm_vec[2]))

            cd_mean = float(np.mean(cds))
            cl_mean = float(np.mean(cls))
            cy_mean = float(np.mean(cys))
            cmx_mean = float(np.mean(cmx_list))
            cmy_mean = float(np.mean(cmy_list))
            cmz_mean = float(np.mean(cmz_list))

            qoi_values["C_D"].append(cd_mean)
            qoi_values["C_D2"].append(cd_mean * cd_mean if np.isfinite(cd_mean) else float("nan"))
            qoi_values["C_L"].append(cl_mean)
            qoi_values["C_Y"].append(cy_mean)
            qoi_values["C_Mx"].append(cmx_mean)
            qoi_values["C_My"].append(cmy_mean)
            qoi_values["C_Mz"].append(cmz_mean)

            with open(os.path.join(subdir, "cpu_time.txt")) as f:
                cpu_time_ms = float(f.read().strip())
            cpu_hours_list.append(cpu_time_ms * self.mpi_procs / 3_600_000.0)

        return qoi_values, cpu_hours_list

    def submit_batch_jobs(
        self,
        altitude,
        AoS,
        db_indices,
        random_seeds=None,
        env_payload_paths=None,
        env_model=None,
        aos_values=None,
        aoa_values=None,
        geometry_id=None,
        geometry_mesh=None,
        flow_zero_direction=None,
    ):
        job_ids = []
        job_subdirs = []
        aos_seq = _expand_values(aos_values if aos_values is not None else AoS, len(db_indices), float(AoS))
        aoa_seq = _expand_values(aoa_values if aoa_values is not None else 0.0, len(db_indices), 0.0)
        seed_seq = _expand_int_values(random_seeds if random_seeds is not None else 1, len(db_indices), 1)
        env_payload_paths = list(env_payload_paths) if env_payload_paths is not None else [None] * len(db_indices)

        group_subdirs = []
        group_db_indices = []
        for pos, db_index in enumerate(db_indices):
            env_payload_path = env_payload_paths[pos] if pos < len(env_payload_paths) else None
            job_subdir = self.prepare_simulation_folder(
                altitude,
                aos_seq[pos],
                db_index,
                random_seed=seed_seq[pos],
                env_payload_path=env_payload_path,
                env_model=env_model,
                geometry_id=geometry_id,
                geometry_mesh=geometry_mesh,
            )
            job_subdirs.append(job_subdir)
            group_subdirs.append(job_subdir)
            group_db_indices.append(db_index)

            if len(group_subdirs) >= self.submission_group_size or pos == len(db_indices) - 1:
                if len(group_subdirs) == 1:
                    job_script = self.create_simulation_job_script(group_subdirs[0])
                else:
                    job_script = self.create_simulation_group_job_script(group_subdirs)
                if self.submit_sleep_s > 0.0:
                    time.sleep(self.submit_sleep_s)
                job_id = self.submit_simulation_job(job_script)
                print(
                    f"Job {job_id} für db_indices {group_db_indices[0]}..{group_db_indices[-1]} "
                    f"({len(group_db_indices)} cases) erstellt und gestartet."
                )
                job_ids.append(job_id)
                group_subdirs = []
                group_db_indices = []

        return {
            "job_ids": job_ids,
            "job_subdirs": job_subdirs,
            "aos_seq": aos_seq,
            "aoa_seq": aoa_seq,
            "flow_zero_direction": flow_zero_direction if flow_zero_direction is not None else self.flow_zero_direction,
            "random_seeds": seed_seq,
            "db_indices": list(db_indices),
        }

    def wait_for_batch_jobs(self, batch_handle, max_retries=2):
        job_ids = list(batch_handle.get("job_ids", []))
        job_subdirs = list(batch_handle.get("job_subdirs", []))
        self._wait_for_jobs_and_retry_failed_outputs(job_subdirs, job_ids, max_retries=max_retries)
        return batch_handle

    def submit_batch_postprocessing(self, batch_handles, random_seed, wait_for_completion=True):
        handles = [batch_handles] if isinstance(batch_handles, dict) else list(batch_handles)
        job_subdirs = []
        for handle in handles:
            job_subdirs.extend(list(handle.get("job_subdirs", [])))
        job_id = self.submit_postprocessing_job(
            job_subdirs,
            random_seed,
            wait_for_completion=wait_for_completion,
        )
        return {
            "job_id": job_id,
            "job_subdirs": job_subdirs,
            "random_seed": int(random_seed),
        }

    def wait_for_postprocessing(self, postprocess_handle):
        job_id = str(postprocess_handle.get("job_id", ""))
        if job_id:
            self._wait_for_job_completion(job_id)
        return postprocess_handle

    def collect_batch_results(self, batch_handle, requested_qois=None):
        job_subdirs = list(batch_handle.get("job_subdirs", []))
        aos_seq = list(batch_handle.get("aos_seq", []))
        aoa_seq = list(batch_handle.get("aoa_seq", []))
        flow_zero_direction = batch_handle.get("flow_zero_direction", self.flow_zero_direction)

        if requested_qois is not None:
            qoi_values, cpu_hours_list = self.collect_results_qois(
                job_subdirs,
                AoS=aos_seq,
                AoA=aoa_seq,
                flow_zero_direction=flow_zero_direction,
            )
            requested = list(requested_qois)
            qoi_values = {q: qoi_values.get(q, [float("nan")] * len(job_subdirs)) for q in requested}
            return qoi_values, np.array(cpu_hours_list)

        mean_cd_list, cpu_hours_list = self.collect_results(job_subdirs)
        return np.array(mean_cd_list), np.array(cpu_hours_list)

    def complete_batch(self, batch_handle, random_seed, requested_qois=None):
        self.wait_for_batch_jobs(batch_handle, max_retries=2)
        self.submit_batch_postprocessing(batch_handle, random_seed, wait_for_completion=True)
        return self.collect_batch_results(batch_handle, requested_qois=requested_qois)

    def run_batch(
        self,
        altitude,
        AoS,
        db_indices,
        random_seed,
        env_payload_paths=None,
        env_model=None,
        aos_values=None,
        aoa_values=None,
        geometry_id=None,
        geometry_mesh=None,
        flow_zero_direction=None,
    ):
        batch_handle = self.submit_batch_jobs(
            altitude,
            AoS,
            db_indices,
            env_payload_paths=env_payload_paths,
            env_model=env_model,
            aos_values=aos_values,
            aoa_values=aoa_values,
            geometry_id=geometry_id,
            geometry_mesh=geometry_mesh,
            flow_zero_direction=flow_zero_direction,
        )
        mean_cd_list, cpu_hours_list = self.complete_batch(batch_handle, random_seed)
        return mean_cd_list, cpu_hours_list

    def run_batch_qois(
        self,
        altitude,
        AoS,
        db_indices,
        random_seed,
        requested_qois=None,
        env_payload_paths=None,
        env_model=None,
        aos_values=None,
        aoa_values=None,
        geometry_id=None,
        geometry_mesh=None,
        flow_zero_direction=None,
    ):
        batch_handle = self.submit_batch_jobs(
            altitude,
            AoS,
            db_indices,
            env_payload_paths=env_payload_paths,
            env_model=env_model,
            aos_values=aos_values,
            aoa_values=aoa_values,
            geometry_id=geometry_id,
            geometry_mesh=geometry_mesh,
            flow_zero_direction=flow_zero_direction,
        )
        qoi_values, cpu_hours_list = self.complete_batch(
            batch_handle,
            random_seed,
            requested_qois=requested_qois,
        )
        return qoi_values, np.array(cpu_hours_list)
