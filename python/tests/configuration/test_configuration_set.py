import copy

import numpy as np
import pytest

import libcasm.configuration as config
import libcasm.xtal as xtal


def test_ConfigurationSet_constructor_1(simple_cubic_binary_prim):
    config.Prim(simple_cubic_binary_prim)
    configurations = config.ConfigurationSet()

    assert isinstance(configurations, config.ConfigurationSet)
    assert configurations.empty()


def test_ConfigurationSet_add_remove_discard_get_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    configurations = config.ConfigurationSet()

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.make_canonical_supercell(config.Supercell(prim, T))
    configuration = config.Configuration(supercell)

    # add remove
    record = configurations.add(configuration)

    assert len(configurations) == 1
    assert isinstance(record, config.ConfigurationRecord)
    assert isinstance(record.configuration, config.Configuration)
    assert record.configuration_id == "0"
    assert record.supercell_name == "SCEL2_1_2_1_1_0_0"
    assert record.configuration_name == "SCEL2_1_2_1_1_0_0/0"
    assert record.configuration == configuration
    assert record.configuration is not configuration

    configurations.remove(configuration)
    assert len(configurations) == 0

    with pytest.raises(KeyError):
        configurations.remove(configuration)

    # add discard
    record = configurations.add(configuration)
    assert len(configurations) == 1

    configurations.discard(configuration)
    assert len(configurations) == 0

    configurations.discard(configuration)
    assert len(configurations) == 0

    # add get
    record_1 = configurations.add(configuration)
    assert len(configurations) == 1

    record_2 = configurations.get(configuration)
    assert record_1 == record_2

    record_3 = configurations.get(record_1.configuration_name)
    assert record_1 == record_3

    configuration_name = copy.copy(record_1.configuration_name)
    configurations.remove(configuration_name)

    record_4 = configurations.get(configuration)
    assert record_4 is None

    record_5 = configurations.get(configuration_name)
    assert record_5 is None


def test_ConfigurationSet_to_dict_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    configurations = config.ConfigurationSet()

    T = np.array(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.make_canonical_supercell(config.Supercell(prim, T))
    configuration = config.Configuration(supercell)

    # add configurations
    for i in range(supercell.n_unitcells):
        print(f"{i}:")
        x = copy.copy(configuration)
        x.set_occ(i, 1)
        print(xtal.pretty_json(x.to_dict(write_prim_basis=False)))
        print(xtal.pretty_json(x.to_dict(write_prim_basis=True)))
        configurations.add(x)
        print()
    assert len(configurations) == 4

    # output to JSON data
    data = configurations.to_dict()
    print(xtal.pretty_json(data))

    # clear
    configurations.clear()
    assert len(configurations) == 0

    # read back in
    supercells = config.SupercellSet(prim)
    configurations_in = config.ConfigurationSet.from_dict(data, supercells)
    assert len(configurations_in) == 4


def test_ConfigurationSet_to_dict_2(FCC_binary_Hstrain_noshear_prim):
    prim = config.Prim(FCC_binary_Hstrain_noshear_prim)
    configurations = config.ConfigurationSet()

    T = np.array(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.make_canonical_supercell(config.Supercell(prim, T))
    configuration = config.Configuration(supercell)

    for i in range(supercell.n_unitcells):
        x = copy.copy(configuration)
        x.set_occ(i, 1)
        configurations.add(x)
    assert len(configurations) == 4

    configurations.clear()
    assert len(configurations) == 0

    for i in range(supercell.n_unitcells):
        x = copy.copy(configuration)
        x.set_occ(i, 1)
        configurations.add(config.make_canonical_configuration(x))
    assert len(configurations) == 1


def test_ConfigurationSet_from_dict_duplicate_warning(simple_cubic_binary_prim):
    """from_dict warns when the input dict has duplicate configurations."""
    prim = config.Prim(simple_cubic_binary_prim)
    supercell = config.make_canonical_supercell(
        config.Supercell(prim, np.eye(3, dtype=int))
    )
    configuration = config.Configuration(supercell)

    configurations = config.ConfigurationSet()
    configurations.add(configuration)
    data = configurations.to_dict()

    # Inject a duplicate entry under a different configuration_id
    scel_name = list(data["supercells"].keys())[0]
    data["supercells"][scel_name]["1"] = data["supercells"][scel_name]["0"]

    supercells = config.SupercellSet(prim)
    with pytest.warns(UserWarning, match="from_dict"):
        configurations_in = config.ConfigurationSet.from_dict(data, supercells)

    assert len(configurations_in) == 1


def test_ConfigurationSet_add_record_overwrite_warning(simple_cubic_binary_prim):
    """add_record warns and overwrites when the same configuration_name is
    inserted with different DoF."""
    prim = config.Prim(simple_cubic_binary_prim)
    supercell = config.make_canonical_supercell(
        config.Supercell(prim, np.eye(3, dtype=int))
    )
    configuration_a = config.Configuration(supercell)
    configuration_b = config.Configuration(supercell)
    configuration_b.set_occ(0, 1)

    configurations = config.ConfigurationSet()
    record_a = configurations.add(configuration_a)

    # Build a record with the same configuration_name but different DoF
    record_b = config.ConfigurationRecord(
        configuration_b,
        record_a.supercell_name,
        record_a.configuration_id,
    )
    assert record_a.configuration_name == record_b.configuration_name
    assert record_a.configuration != record_b.configuration

    with pytest.warns(UserWarning, match="add_record"):
        configurations.add_record(record_b)

    assert len(configurations) == 1
    result = configurations.get(record_a.configuration_name)
    assert result is not None
    assert result.configuration == configuration_b


def test_Configuration_ConfigurationRecord_comparison_warning(
    simple_cubic_binary_prim,
):
    """Comparing a Configuration with a ConfigurationRecord warns."""
    prim = config.Prim(simple_cubic_binary_prim)
    supercell = config.make_canonical_supercell(
        config.Supercell(prim, np.eye(3, dtype=int))
    )
    configuration = config.Configuration(supercell)

    configurations = config.ConfigurationSet()
    record = configurations.add(configuration)

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = configuration == record
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = record == configuration
    assert result is False

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = configuration != record
    assert result is True

    with pytest.warns(UserWarning, match="ConfigurationRecord"):
        result = record != configuration
    assert result is True


def test_ConfigurationRecord_repr(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    configurations = config.ConfigurationSet()

    T = np.array(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.make_canonical_supercell(config.Supercell(prim, T))
    configuration = config.Configuration(supercell)

    # add configurations
    for i in range(supercell.n_unitcells):
        x = configuration.copy()
        x.set_occ(i, 1)
        configurations.add(x)
    assert len(configurations) == 4

    # Test print ConfigurationRecord
    import io
    from contextlib import redirect_stdout

    for record in configurations:
        assert isinstance(record, config.ConfigurationRecord)
        f = io.StringIO()
        with redirect_stdout(f):
            print(record)
        out = f.getvalue()
        assert "configuration" in out
        assert "supercell_name" in out
        assert "configuration_id" in out
        assert "configuration_name" in out
