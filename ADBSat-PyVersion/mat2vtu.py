import scipy.io
import numpy as np
import meshio
import os

def mat2vtu(mesh_file, results_file, scalar_key='cd', output_filename=None):
    """
    Konvertiert ADBSat-Mesh- und Ergebnisdaten in eine VTU-Datei mit meshio und gibt den Mittelwert des Skalarfelds zurück.

    Args:
        mesh_file (str): Pfad zur .mat Datei mit 'meshdata'
        results_file (str): Pfad zur .mat Datei mit Ergebnisdaten (z. B. 'cd')
        scalar_key (str): Key im Ergebnisfile für das gewünschte Skalarfeld
        output_filename (str, optional): Pfad zur Ausgabedatei (.vtu)

    Returns:
        float: Mittelwert des Skalarfelds
    """

    # Konstanten & Feldnamen
    mesh_main_key = 'meshdata'
    x_field, y_field, z_field = 'XData', 'YData', 'ZData'
    POINTS_PER_CELL = 3  # für Dreiecke

    # Lade Daten
    mesh_data = scipy.io.loadmat(mesh_file)
    results_data = scipy.io.loadmat(results_file)

    scalar_data = results_data[scalar_key].flatten()
    num_faces = len(scalar_data)

    # Extrahiere Koordinaten
    mesh_struct = mesh_data[mesh_main_key][0, 0]
    x_all = mesh_struct[x_field][:, :num_faces]
    y_all = mesh_struct[y_field][:, :num_faces]
    z_all = mesh_struct[z_field][:, :num_faces]

    # Alle 3 Knotenpunkte pro Dreieck stapeln
    all_vertices = np.vstack([
        np.column_stack((x_all[i, :], y_all[i, :], z_all[i, :]))
        for i in range(POINTS_PER_CELL)
    ])

    # Eindeutige Punkte bestimmen
    unique_verts, inverse_map = np.unique(all_vertices, axis=0, return_inverse=True)
    faces = inverse_map.reshape(POINTS_PER_CELL, num_faces).T

    '''    # Speichere VTU-Datei
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        output_filename = f"{base_name}_{scalar_key}.vtu"

    meshio.write_points_cells(
        output_filename,
        points=unique_verts,
        cells=[("triangle", faces)],
        cell_data={scalar_key: [scalar_data]}
    )
   '''

    return float(np.mean(scalar_data))

