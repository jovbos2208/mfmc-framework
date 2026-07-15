import numpy as np
from .shadowAnaly import shadowAnaly
from .coeff_solar import coeff_solar
from .mainCoeff import mainCoeff
from scipy.io import loadmat, savemat
import os


def _mat_scalar(value):
    arr = np.asarray(value)
    while arr.size == 1 and arr.dtype == object:
        arr = np.asarray(arr.item())
    return float(np.asarray(arr, dtype=float).reshape(-1)[0])


def calc_coeff(fi_name, respath, aoaS, aosS, param_eq, flag_shad, flag_sol, dp,
               delete_temp_files: bool = False, verbose: bool = False,
               write_mat: bool = True, return_details: bool = False):
    """
    Calculates local and global aerodynamic (and, optionally, solar–radiation‑pressure)
    coefficients for a triangular‑mesh geometry.

    Compared with the original MATLAB implementation this Python version now also
    computes the global force and moment coefficients **after** the optional shadow
    analysis – exactly like the historic “calc_coeff.m”.  All coefficients are written
    to the resulting *.mat* file so that downstream post‑processing scripts do not
    lose any information.

    Parameters
    ----------
    fi_name : str
        Path to the *.mat* file that contains the *meshdata* structure.
    respath : str
        Folder where result files will be written.
    aoaS, aosS : 1‑d array‑like of float (rad)
        Angles of attack and side‑slip that shall be evaluated.
    param_eq : dict
        Parameter dictionary for the selected GSI model. *gamma* and *ell* are
        overwritten internally – do **not** supply them.
    flag_shad : bool
        If *True* the **shadowAnaly** routine is executed.
    flag_sol : bool
        If *True* the solar radiation pressure terms are computed together with
        the aerodynamic ones.
    dp : Any
        Dummy place‑holder to keep the original call signature intact.
    delete_temp_files, verbose : bool
        Optional housekeeping & logging flags.

    Returns
    -------
    str
        Full path of the last written *.mat* result file.
    """

    # -------------------------------------------------------------------------
    # 1) Load geometry from the input *.mat* file
    # -------------------------------------------------------------------------
    mat_data = loadmat(fi_name)
    meshdata = mat_data['meshdata']

    x = meshdata['XData'][0, 0]         # (N,) panel‑vertex x‑coordinates
    y = meshdata['YData'][0, 0]         # (N,) panel‑vertex y‑coordinates
    z = meshdata['ZData'][0, 0]         # (N,) panel‑vertex z‑coordinates
    areas = meshdata['Areas'][0, 0].ravel()   # (N,) panel areas
    surfN = meshdata['SurfN'][0, 0]     # (3, N) surface normals
    barC = meshdata['BariC'][0, 0]      # (3, N) panel centroids
    len_ref = _mat_scalar(meshdata['Lref'][0, 0])  # scalar reference length
    matID = meshdata['MatID'][0, 0].ravel()  # (N,) material IDs per panel

    # -------------------------------------------------------------------------
    # 2) Some pre‑processing helpers
    # -------------------------------------------------------------------------
    def _repeat(v, n):
        """Return *v* tiled *n* times as column‑matrix (shape 3×N)."""
        return np.tile(v.reshape(3, 1), n)

    idx_aoa = len(aoaS)
    idx_aos = len(aosS)

    # Prepare output directory – allow for simple batch processing
    mat_name = os.path.splitext(os.path.basename(fi_name))[0]
    if idx_aoa * idx_aos > 1:
        folder_name = f"{mat_name}_{np.random.randint(1, 1000):03d}"
        os.makedirs(os.path.join(respath, folder_name), exist_ok=True)
        path_res = os.path.join(respath, folder_name)
    else:
        path_res = respath

    # -------------------------------------------------------------------------
    # 3) Loop over all AoA/AoS combinations
    # -------------------------------------------------------------------------
    last_details = None
    for aoa in aoaS:
        for aos in aosS:

            # 3.1) Direction‑cosine matrices (wind‑>body, ground‑>body …)
            L_wb = np.array([
                [np.cos(aos) * np.cos(aoa),  np.sin(aos),              np.sin(aoa) * np.cos(aos)],
                [-np.sin(aos) * np.cos(aoa), np.cos(aos),             -np.sin(aoa) * np.sin(aos)],
                [-np.sin(aoa),               0.0,                      np.cos(aoa)]
            ])

            L_gb = np.diag([1.0, -1.0, -1.0])          # ground –> body
            L_gw = L_gb @ L_wb.T                       # ground –> wind
            L_fb = np.diag([-1.0, 1.0, -1.0])          # flow –> body

            # 3.2) Flow direction expressed in *wind* frame
            v_dir = L_gw @ np.array([-1.0, 0.0, 0.0])   # unit flow vector
           
            v_dir /= np.linalg.norm(v_dir)
            print(v_dir)
            v_matrix = _repeat(v_dir, surfN.shape[1])

            # 3.3) Angle between panel normal and flow direction
            delta = np.arccos(np.einsum('ij,ij->j', -v_matrix, surfN))

            # -----------------------------------------------------------------
            # 3.4) Populate *param_eq* for **mainCoeff** (γ & ℓ depend on *δ*)
            # -----------------------------------------------------------------
            param_eq = param_eq.copy()  # do **not** modify caller's dict!
            param_eq["gamma"] = np.cos(delta)
            param_eq["ell"]   = np.sin(delta)

            # 3.5) Local aerodynamic panel coefficients (cp, ctau, cd, cl)
            cp, ctau, cd, cl = mainCoeff(param_eq, delta, matID)

            # 3.6) Optional solar pressure panel coefficients (cn, cs)
            if flag_sol:
                cn, cs = coeff_solar(delta, param_eq)

            # -----------------------------------------------------------------
            # 3.7) *Shadow* analysis – panels in the wake yield zero contribution
            # -----------------------------------------------------------------
            if flag_shad:
                shad_pan = shadowAnaly(x, y, z, barC, delta, L_gw)
                cp[shad_pan]   = 0.0
                ctau[shad_pan] = 0.0
                cd[shad_pan]   = 0.0
                cl[shad_pan]   = 0.0
                if flag_sol:
                    cn[shad_pan] = 0.0
                    cs[shad_pan] = 0.0

            # =================================================================
            # 3.8) ***Global*** force & moment coefficients (NEW)
            # =================================================================
            # Use the same wind-projected reference area convention as the
            # PICLas postprocessor: 0.5 * sum(|n_i . flow| * A_i).
            area_proj = 0.5 * np.abs(areas * np.cos(delta))
            
            area_total = float(np.sum(areas))
            area_proj_total = float(np.sum(area_proj))
            area_ref = max(area_proj_total, 1e-12)
            area_ref_source = "adbsat_wind_projected"

            # --- shear direction ------------------------------------------------
            tau_dir = np.cross(surfN.T, np.cross(v_matrix.T, surfN.T)).T  # 3×N
            # normalise & handle edge cases (δ = 90° ⇒ |τ| → 0)
            tau_norm = np.linalg.norm(tau_dir, axis=0)
            good = tau_norm > 0.0
            tau_dir[:, good] /= tau_norm[good]
            tau_dir[:, ~good] = 0.0

            # --- force coefficients in *ground* axes ---------------------------
            ctau_area = ctau * areas        # (N,)
            cp_area   = cp   * areas

            Cf_g = (tau_dir @ ctau_area - surfN @ cp_area) / area_ref  # (3,)
            panel_traction_over_q = (tau_dir * ctau.reshape(1, -1) - surfN * cp.reshape(1, -1)).T
            panel_force_per_area = float(dp) * panel_traction_over_q
            reconstructed_cf_g = np.sum(panel_traction_over_q * areas[:, None], axis=0) / area_ref
            if not np.allclose(reconstructed_cf_g, Cf_g, rtol=1.0e-12, atol=1.0e-14):
                raise RuntimeError("ADBSat panel traction does not reconstruct the integrated force coefficient")
            # wind‑, flow‑aligned and body coefficients
            Cf_w = L_gw.T @ Cf_g
            Cf_f = L_fb @ (L_gb.T @ Cf_g)

            # --- moment coefficients in *ground* axes --------------------------
            cross_barC_tau = np.cross(barC.T,  tau_dir.T).T
            cross_barC_nrm = np.cross(barC.T, -surfN.T).T
            Cm_g = (cross_barC_tau @ ctau_area + cross_barC_nrm @ cp_area)
            Cm_g /= (area_ref * len_ref)                    # (3,)
            Cm_b = L_gb.T @ Cm_g                            # body axes

            # --- solar‑pressure contributions (optional) -----------------------
            if flag_sol:
                cstau = cs * np.sin(delta)          # incident shear component
                cs_n  = cs * np.cos(delta)          # incident normal component

                cstau_area = cstau * areas
                csn_area   = (cn + cs_n) * areas

                Cs_g = (tau_dir @ cstau_area - surfN @ csn_area) / area_ref
                Cf_s = L_gw.T @ Cs_g

                cross_barC_tau_s = cross_barC_tau   # τ‑direction unchanged
                cross_barC_nrm_s = cross_barC_nrm   # n‑direction unchanged

                Cm_s_g = (cross_barC_tau_s @ cstau_area + cross_barC_nrm_s @ csn_area)
                Cm_s_g /= (area_ref * len_ref)
                Cm_s_b = L_gb.T @ Cm_s_g
            else:
                Cs_g = Cf_s = Cm_s_g = Cm_s_b = np.zeros(3)

            # -----------------------------------------------------------------
            # 3.9) Write everything to *.mat* (one file per AoA/AoS combination)
            # -----------------------------------------------------------------
            method = param_eq['gsi_model']
            fn_out = f"{mat_name}_{method}_aoa{int(np.degrees(aoa))}_aos{int(np.degrees(aos))}.mat"
            path_out = os.path.join(path_res, fn_out)

            if write_mat:
                savemat(path_out, {
                    # local panel coefficients ------------------------------------
                    'cp': cp, 'ctau': ctau, 'cd': cd, 'cl': cl,
                    'delta': delta,
                    # global aerodynamic coefficients -----------------------------
                    'Cf_g': Cf_g, 'Cf_w': Cf_w, 'Cf_f': Cf_f,
                    'Cm_G': Cm_g, 'Cm_B': Cm_b,
                    # solar‑pressure coefficients (may be zeros) ------------------
                    'Cs_G': Cs_g, 'Cf_s': Cf_s, 'Cm_s_G': Cm_s_g, 'Cm_s_B': Cm_s_b,
                    # misc meta‑data ----------------------------------------------
                    'aoa': aoa, 'aos': aos,
                    'AreaRef': area_ref,
                    'AreaTotal': area_total,
                    'AreaProj': area_proj,
                    'AreaProjTotal': area_proj_total,
                    'AreaRefSource': area_ref_source,
                    'LenRef': len_ref
                })

            if verbose:
                print(f"[calc_coeff] AoA = {np.degrees(aoa):5.1f}°, AoS = {np.degrees(aos):5.1f}°  →  {fn_out}")

            last_path = path_out  # keep track of the very last file
            last_details = {
                "C_D": float(abs(-Cf_w[0])),
                "C_L": float(Cf_w[2]),
                "C_Y": float(Cf_w[1]),
                "C_Mx": float(Cm_b[0]),
                "C_My": float(Cm_b[1]),
                "C_Mz": float(Cm_b[2]),
                "AreaRef": float(area_ref),
                "AreaTotal": float(area_total),
                "AreaProjTotal": float(area_proj_total),
                "AreaRefSource": area_ref_source,
                "panel_traction_over_q": panel_traction_over_q,
                "panel_force_per_area": panel_force_per_area,
                "panel_area": areas.copy(),
                "panel_center": barC.T.copy(),
                "panel_normal": surfN.T.copy(),
                "u_hat_inf": v_dir.copy(),
                "q_inf": float(dp),
            }

    # -------------------------------------------------------------------------
    # 4) Optional: clean‑up of temporaries – currently a no‑op but kept for API
    # -------------------------------------------------------------------------
    if delete_temp_files:
        # placeholder – implement if/when temporary artefacts are generated
        pass

    if return_details and last_details is not None:
        return last_details
    return abs(-Cf_w[0])

