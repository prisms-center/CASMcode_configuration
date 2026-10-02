import numpy as np
import pytest

import libcasm.configuration as casmconfig
import libcasm.configuration.io.spglib as spglib_io

from .functions import check_symmetry_dataset


def permute_columns(perm, before):
    # after[i] = before[permutation[i]];
    after = np.zeros(before.shape)
    for i in range(len(perm)):
        after[:, i] = before[:, perm[i]]
    return after


def test_configuration_with_properties_1a(FCC_binary_prim):
    """Construct with configuration and properties"""
    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)

    # T_prim = np.eye(3, dtype=int)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, T_conventional)
    )
    config_init = casmconfig.Configuration(
        supercell=supercell,
    )

    occupation_init = np.array([1, 1, 0, 0], dtype=int)
    config_init.set_occupation(occupation_init)

    disp_init = np.array(
        [
            [0.01, 0.05, 0.09],
            [0.02, 0.06, 0.10],
            [0.03, 0.07, 0.11],
            [0.04, 0.08, 0.12],
        ]
    ).transpose()
    local_properties_init = {
        "disp": disp_init,
    }
    energy_scalar_init = 0.1
    energy_init = np.array([energy_scalar_init])
    Hstrain_init = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    global_properties_init = {
        "energy": energy_init,
        "Hstrain": Hstrain_init,
    }
    config_w_props = casmconfig.ConfigurationWithProperties(
        configuration=config_init,
        local_properties=local_properties_init,
        global_properties=global_properties_init,
    )

    assert "disp" in config_w_props.local_properties
    assert np.allclose(config_w_props.local_properties["disp"], disp_init)
    assert np.allclose(config_w_props.local_property_values("disp"), disp_init)

    assert "energy" in config_w_props.global_properties
    assert np.allclose(config_w_props.global_properties["energy"], energy_init)
    assert np.allclose(config_w_props.global_property_values("energy"), energy_init)
    assert np.isclose(
        config_w_props.scalar_global_property_value("energy"), energy_scalar_init
    )

    assert "Hstrain" in config_w_props.global_properties
    assert np.allclose(config_w_props.global_properties["Hstrain"], Hstrain_init)
    assert np.allclose(config_w_props.global_property_values("Hstrain"), Hstrain_init)


def test_configuration_with_properties_1b(FCC_binary_prim):
    """Construct with configuration only, then set properties"""
    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)

    # T_prim = np.eye(3, dtype=int)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, T_conventional)
    )
    config_init = casmconfig.Configuration(
        supercell=supercell,
    )

    occupation_init = np.array([1, 1, 0, 0], dtype=int)
    config_init.set_occupation(occupation_init)

    config_w_props = casmconfig.ConfigurationWithProperties(
        configuration=config_init,
    )

    assert len(config_w_props.local_properties) == 0
    assert len(config_w_props.global_properties) == 0


def test_configuration_with_properties_2(FCC_binary_prim):
    """Apply SupercellSymOp"""
    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)

    # T_prim = np.eye(3, dtype=int)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, T_conventional)
    )
    config_init = casmconfig.Configuration(
        supercell=supercell,
    )

    occupation_init = np.array([1, 1, 0, 0], dtype=int)
    config_init.set_occupation(occupation_init)
    config_factor_group = casmconfig.make_invariant_subgroup(
        configuration=config_init,
    )

    disp_init = np.array(
        [
            [0.01, 0.05, 0.09],
            [0.02, 0.06, 0.10],
            [0.03, 0.07, 0.11],
            [0.04, 0.08, 0.12],
        ]
    ).transpose()
    local_properties_init = {
        "disp": disp_init,
    }
    energy_init = np.array([[0.1]]).transpose()
    Hstrain_init = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]).transpose()
    global_properties_init = {
        "energy": energy_init,
        "Hstrain": Hstrain_init,
    }
    config_w_props = casmconfig.ConfigurationWithProperties(
        configuration=config_init,
        local_properties=local_properties_init,
        global_properties=global_properties_init,
    )

    # print("~~~ config w/ props initial ~~~")
    # print(xtal.pretty_json(config_w_props.to_dict()))
    assert (config_w_props.configuration.occupation == occupation_init).all()
    # print()

    i = 0
    for supercell_symop in config_factor_group:
        transformed = supercell_symop * config_w_props

        # print(f"~~~ config w/ props transformed {i} ~~~")
        # print(xtal.pretty_json(transformed.to_dict()))
        assert (transformed.configuration.occupation == occupation_init).all()
        i += 1

        ### Test consistency with applying xtal.SymOp ###
        symop = supercell_symop.to_symop()
        perm = supercell_symop.combined_permute()

        # disp
        transformed_local_properties = symop * local_properties_init
        assert transformed.local_properties["disp"].shape == (3, 4)
        assert np.allclose(
            transformed.local_properties["disp"],
            permute_columns(perm, transformed_local_properties["disp"]),
        )
        assert np.allclose(
            transformed.local_properties["disp"],
            permute_columns(perm, symop.matrix() @ disp_init),
        )
        assert np.allclose(
            transformed.local_properties["disp"],
            permute_columns(perm, symop.matrix_rep("disp") @ disp_init),
        )

        # energy
        transformed_global_properties = symop * global_properties_init
        assert transformed.global_properties["energy"].shape == (1,)
        assert np.allclose(
            transformed.global_properties["energy"],
            transformed_global_properties["energy"].flatten(),
        )
        assert np.allclose(
            transformed.global_properties["energy"],
            (symop.matrix_rep("energy") @ energy_init).flatten(),
        )
        assert np.isclose(
            transformed.scalar_global_property_value("energy"), energy_init[0]
        )

        # Hstrain
        assert transformed.global_properties["Hstrain"].shape == (6,)
        assert np.allclose(
            transformed.global_properties["Hstrain"],
            transformed_global_properties["Hstrain"].flatten(),
        )
        assert np.allclose(
            transformed.global_properties["Hstrain"],
            (symop.matrix_rep("Hstrain") @ Hstrain_init).flatten(),
        )

        # print()


def make_config_with_props(
    prim: casmconfig.Prim,
    T: np.ndarray,
    occupation: np.ndarray,
    disp: np.ndarray,
    Hstrain: np.ndarray,
):
    supercell = casmconfig.make_canonical_supercell(casmconfig.Supercell(prim, T))
    config_init = casmconfig.Configuration(
        supercell=supercell,
    )

    config_init.set_occupation(occupation)
    local_properties_init = {
        "disp": disp,
    }
    global_properties_init = {
        "Hstrain": Hstrain,
    }
    return casmconfig.ConfigurationWithProperties(
        configuration=config_init,
        local_properties=local_properties_init,
        global_properties=global_properties_init,
    )


def test_config_with_props_spglib_io_1(FCC_binary_prim):
    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    occupation = np.array([1, 1, 0, 0], dtype=int)
    disp = np.array(
        [
            [0.01, 0.05, 0.09],
            [0.02, 0.06, 0.10],
            [0.03, 0.07, 0.11],
            [0.04, 0.08, 0.12],
        ]
    ).transpose()
    Hstrain = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]).transpose()
    config_w_props = make_config_with_props(
        prim=prim,
        T=T_conventional,
        occupation=occupation,
        disp=disp,
        Hstrain=Hstrain,
    )

    ###

    cell = spglib_io.as_cell(config_w_props)

    assert isinstance(cell, tuple)
    assert len(cell) == 3

    symmetry_dataset = spglib_io.get_symmetry_dataset(config_w_props)
    check_symmetry_dataset(symmetry_dataset, number=1, n_rotations=1)


def test_config_with_props_spglib_io_2(FCC_binary_prim):
    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    occupation = np.array([0, 0, 0, 0], dtype=int)
    disp = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ]
    ).transpose()
    Hstrain = np.array([[0.01, 0.01, 0.01, 0.00, 0.00, 0.00]]).transpose()
    config_w_props = make_config_with_props(
        prim=prim,
        T=T_conventional,
        occupation=occupation,
        disp=disp,
        Hstrain=Hstrain,
    )

    ###

    cell = spglib_io.as_cell(config_w_props)

    assert isinstance(cell, tuple)
    assert len(cell) == 3

    symmetry_dataset = spglib_io.get_symmetry_dataset(config_w_props)
    check_symmetry_dataset(symmetry_dataset, number=225, n_rotations=4 * 48)


def test_configuration_with_properties_io(FCC_binary_prim):
    """Apply SupercellSymOp"""
    import io
    from contextlib import redirect_stdout

    xtal_prim = FCC_binary_prim
    prim = casmconfig.Prim(xtal_prim)

    # T_prim = np.eye(3, dtype=int)
    T_conventional = np.array(
        [  # conventional FCC cubic cell
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, T_conventional)
    )
    config_init = casmconfig.Configuration(
        supercell=supercell,
    )

    occupation_init = np.array([1, 1, 0, 0], dtype=int)
    config_init.set_occupation(occupation_init)

    disp_init = np.array(
        [
            [0.01, 0.05, 0.09],
            [0.02, 0.06, 0.10],
            [0.03, 0.07, 0.11],
            [0.04, 0.08, 0.12],
        ]
    ).transpose()
    local_properties_init = {
        "disp": disp_init,
    }
    energy_init = np.array([[0.1]]).transpose()
    Hstrain_init = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]).transpose()
    global_properties_init = {
        "energy": energy_init,
        "Hstrain": Hstrain_init,
    }
    config_w_props = casmconfig.ConfigurationWithProperties(
        configuration=config_init,
        local_properties=local_properties_init,
        global_properties=global_properties_init,
    )

    # Test to_dict (standard basis)
    data = config_w_props.to_dict()
    assert isinstance(data, dict)

    # Test from_dict (standard basis)
    supercell_set = casmconfig.SupercellSet(prim)
    config_w_props_in = casmconfig.ConfigurationWithProperties.from_dict(
        data=data, supercells=supercell_set
    )
    assert isinstance(config_w_props_in, casmconfig.ConfigurationWithProperties)
    assert len(supercell_set) == 1
    assert config_w_props_in.configuration == config_w_props.configuration
    assert np.allclose(
        config_w_props_in.local_properties["disp"],
        config_w_props.local_properties["disp"],
    )
    assert np.allclose(
        config_w_props_in.global_properties["energy"],
        config_w_props.global_properties["energy"],
    )
    assert np.allclose(
        config_w_props_in.global_properties["Hstrain"],
        config_w_props.global_properties["Hstrain"],
    )

    # Test print

    f = io.StringIO()
    with redirect_stdout(f):
        print(config_w_props)
    out = f.getvalue()
    assert "configuration" in out
    assert "global_properties" in out
    assert "local_properties" in out


def test_ConfigurationRecord_ConfigurationWithProperties_comparison_warning(
    FCC_binary_prim,
):
    """Comparing a ConfigurationWithProperties with a ConfigurationRecord warns."""
    prim = casmconfig.Prim(FCC_binary_prim)
    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, np.eye(3, dtype=int))
    )
    configuration = casmconfig.Configuration(supercell)
    configuration_with_properties = casmconfig.ConfigurationWithProperties(
        configuration
    )

    configurations = casmconfig.ConfigurationSet()
    record = configurations.add(configuration)

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = configuration_with_properties == record
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = record == configuration_with_properties
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = configuration_with_properties != record
    assert result is True

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = record != configuration_with_properties
    assert result is True


def test_Configuration_ConfigurationWithProperties_comparison_warning(
    FCC_binary_prim,
):
    """Comparing a Configuration with a ConfigurationWithProperties warns."""
    prim = casmconfig.Prim(FCC_binary_prim)
    supercell = casmconfig.make_canonical_supercell(
        casmconfig.Supercell(prim, np.eye(3, dtype=int))
    )
    configuration = casmconfig.Configuration(supercell)
    configuration_with_properties = casmconfig.ConfigurationWithProperties(
        configuration
    )

    with pytest.warns(UserWarning, match="ConfigurationWithProperties"):
        result = configuration == configuration_with_properties
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationWithProperties"):
        result = configuration_with_properties == configuration
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationWithProperties"):
        result = configuration != configuration_with_properties
    assert result is True

    with pytest.warns(UserWarning, match="ConfigurationWithProperties"):
        result = configuration_with_properties != configuration
    assert result is True
