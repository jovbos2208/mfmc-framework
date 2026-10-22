"""Panel shadow analysis ported from the original MATLAB ADBSat routine."""

import numpy as np

from .insidetri import insidetri


def shadowAnaly(x, y, z, barC, delta, L_gw, surfN):
    """Return zero-based indices of flow-facing panels shadowed by other panels.

    This follows ``ADBSat/toolbox/calc/shadowAnaly.m`` directly. Candidate
    triangles are projected into the wind-frame y-z plane, the projected
    centroid is tested with the MATLAB barycentric point-in-triangle test,
    and the final depth test uses the candidate panel normal.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    barC = np.asarray(barC, dtype=float)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    L_gw = np.asarray(L_gw, dtype=float)
    surfN = np.asarray(surfN, dtype=float)

    pAw = L_gw.T @ np.vstack((x[0, :], y[0, :], z[0, :]))
    pBw = L_gw.T @ np.vstack((x[1, :], y[1, :], z[1, :]))
    pCw = L_gw.T @ np.vstack((x[2, :], y[2, :], z[2, :]))
    barCw = L_gw.T @ barC

    xW = np.vstack((pAw[0, :], pBw[0, :], pCw[0, :]))
    xWmax = np.max(xW, axis=0)
    xWmin = np.min(xW, axis=0)

    delta_deg = np.degrees(delta)
    indB = np.flatnonzero(delta_deg > 90.0001)
    indF = np.flatnonzero(delta_deg <= 90.0001)
    if indB.size == 0 or indF.size == 0:
        return np.empty(0, dtype=int)

    minXwF = np.min(xWmin[indF])
    maxXwB = np.max(xWmin[indB])

    # Preserve the MATLAB expression ``10e-5`` (= 1e-4) exactly.
    candidate_tol = 10e-5
    indFPot = np.flatnonzero(xWmin[indF] - maxXwB < candidate_tol)
    indBPot = np.flatnonzero(xWmax[indB] - minXwF > candidate_tol)
    if indFPot.size == 0 or indBPot.size == 0:
        return np.empty(0, dtype=int)

    potential_back_panels = indB[indBPot]
    yWC = np.vstack((pAw[1, potential_back_panels],
                     pBw[1, potential_back_panels],
                     pCw[1, potential_back_panels]))
    zWC = np.vstack((pAw[2, potential_back_panels],
                     pBw[2, potential_back_panels],
                     pCw[2, potential_back_panels]))

    shadowed = np.zeros(barCw.shape[1], dtype=bool)
    depth_tol = 1e-5

    for front_position in indFPot:
        front_panel = indF[front_position]
        transYcord = yWC - barCw[1, front_panel]
        yCoord = np.abs(np.sum(np.sign(transYcord), axis=0))
        yC_chang = np.flatnonzero(yCoord < 3)
        if yC_chang.size == 0:
            continue

        transZcord = zWC[:, yC_chang] - barCw[2, front_panel]
        zCoord = np.abs(np.sum(np.sign(transZcord), axis=0))
        zC_chang = np.flatnonzero(zCoord < 3)
        if zC_chang.size == 0:
            continue

        # zC_chang indexes the y-filtered subset; map it back before taking
        # y coordinates or resolving global panel indices.
        projected_candidates = yC_chang[zC_chang]
        p1 = np.vstack((transYcord[0, projected_candidates], transZcord[0, zC_chang]))
        p2 = np.vstack((transYcord[1, projected_candidates], transZcord[1, zC_chang]))
        p3 = np.vstack((transYcord[2, projected_candidates], transZcord[2, zC_chang]))

        inside_positions = insidetri(p1, p2, p3, np.zeros((2, zC_chang.size)))
        if inside_positions.size == 0:
            continue

        occluder_positions = projected_candidates[inside_positions]
        occluder_panels = potential_back_panels[occluder_positions]
        normals_w = L_gw.T @ surfN[:, occluder_panels]
        centroid_to_vertex = barCw[:, front_panel, None] - pAw[:, occluder_panels]
        depth = np.sum(normals_w * centroid_to_vertex, axis=0)
        if np.any(depth > depth_tol):
            shadowed[front_panel] = True

    return np.flatnonzero(shadowed)
