from mfmc_campaign.pilot_correlation import _build_metadata


def test_pilot_correlation_forwards_flow_convention_to_model_requests() -> None:
    config = {
        "execution": {
            "aos_deg": 0.0,
            "aoa_deg": 0.0,
            "flow_zero_direction": [1.0, 0.0, 0.0],
            "adbsat_aos_offset_deg": 0.0,
            "environment": {
                "model": "csv",
                "relative_speed_mps": 7718.371,
            },
        }
    }
    geometry = {
        "id": "cylinder_hex_scale_0p1_l1",
        "name": "Cylinder-Hex L1",
        "geometry_class": "parametric_cylinder_hex",
        "metadata": {
            "hf_mesh": "geometry/cylinder_hex_convergence/L1/cylinder_hex_scale_0p1_l1_mesh.h5",
            "piclas_object_boundary_name": "CYLINDER_HEX",
        },
    }

    metadata = _build_metadata(config, geometry, geometry["id"])

    assert metadata["flow_zero_direction"] == [1.0, 0.0, 0.0]
    assert metadata["aos_deg"] == 0.0
    assert metadata["aoa_deg"] == 0.0
    assert metadata["adbsat_aos_offset_deg"] == 0.0
    assert metadata["relative_speed_mps"] == 7718.371
