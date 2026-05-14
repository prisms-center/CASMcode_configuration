import numpy as np
import pytest

import libcasm.configuration as config


def test_SupercellSet_constructor_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)

    assert isinstance(supercells, config.SupercellSet)
    assert supercells.empty()


def test_SupercellSet_add_remove_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)
    assert supercells.empty()

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.Supercell(prim, T)
    assert config.is_canonical_supercell(supercell) is False

    supercell = config.make_canonical_supercell(supercell)
    assert config.is_canonical_supercell(supercell) is True

    # add remove supercell
    record = supercells.add(supercell)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.remove(supercell)
    assert supercells.empty() is True
    assert len(supercells) == 0

    with pytest.raises(KeyError):
        supercells.remove(supercell)

    # add remove T
    supercells.add(T)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.remove(T)
    assert supercells.empty() is True
    assert len(supercells) == 0

    with pytest.raises(KeyError):
        supercells.remove(T)

    # add remove record
    record = config.SupercellRecord(supercell)
    supercells.add(record)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.remove(record)
    assert supercells.empty() is True
    assert len(supercells) == 0

    with pytest.raises(KeyError):
        supercells.remove(record)

    # add remove canonical by name
    record = config.SupercellRecord(supercell)
    supercells.add(record.canonical_supercell_name)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.remove(record.canonical_supercell_name)
    assert supercells.empty() is True
    assert len(supercells) == 0

    with pytest.raises(KeyError):
        supercells.remove(record.canonical_supercell_name)


def test_SupercellSet_add_discard_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)
    assert supercells.empty()

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.Supercell(prim, T)
    assert config.is_canonical_supercell(supercell) is False

    supercell = config.make_canonical_supercell(supercell)
    assert config.is_canonical_supercell(supercell) is True

    # add discard supercell
    record = supercells.add(supercell)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.discard(supercell)
    assert supercells.empty() is True
    assert len(supercells) == 0
    supercells.discard(supercell)

    # add discard T
    supercells.add(T)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.discard(T)
    assert supercells.empty() is True
    assert len(supercells) == 0
    supercells.discard(T)

    # add discard record
    record = config.SupercellRecord(supercell)
    supercells.add(record)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.discard(record)
    assert supercells.empty() is True
    assert len(supercells) == 0
    supercells.discard(record)

    # add discard canonical by name
    record = config.SupercellRecord(supercell)
    supercells.add(record.canonical_supercell_name)
    assert supercells.empty() is False
    assert len(supercells) == 1

    supercells.discard(record.canonical_supercell_name)
    assert supercells.empty() is True
    assert len(supercells) == 0
    supercells.discard(record.canonical_supercell_name)


def test_SupercellSet_to_dict_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)
    assert supercells.empty()

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.Supercell(prim, T)
    assert config.is_canonical_supercell(supercell) is False

    supercell = config.make_canonical_supercell(supercell)
    assert config.is_canonical_supercell(supercell) is True

    # check canonical supercell
    supercells.add(supercell)
    data = supercells.to_dict()
    assert "supercells" in data
    assert len(data["supercells"]) == 1
    assert "non_canonical_supercells" in data
    assert len(data["non_canonical_supercells"]) == 0
    supercells.remove(supercell)

    # check non-canonical supercell
    supercells.add(T)
    data = supercells.to_dict()
    assert "supercells" in data
    assert len(data["supercells"]) == 0
    assert "non_canonical_supercells" in data
    assert len(data["non_canonical_supercells"]) == 1
    supercells.remove(T)


def test_SupercellSet_from_dict_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)

    data = {
        "non_canonical_supercells": [],
        "supercells": {"SCEL2_1_2_1_1_0_0": [[-1, 0, 0], [0, 1, 1], [0, 1, -1]]},
        "version": "2.0",
    }
    supercells = config.SupercellSet.from_dict(data, prim)
    assert isinstance(supercells, config.SupercellSet)
    assert len(supercells) == 1

    data = {
        "non_canonical_supercells": [
            {
                "canonical_supercell_name": "SCEL2_1_2_1_1_0_0",
                "supercell_name": "SCEL2_2_1_1_0_0_1",
                "transformation_matrix_to_supercell": [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
        "supercells": {},
        "version": "2.0",
    }
    supercells = config.SupercellSet.from_dict(data, prim)
    assert isinstance(supercells, config.SupercellSet)
    assert len(supercells) == 1


def test_SupercellSet_to_dict_v1_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)
    assert supercells.empty()

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.Supercell(prim, T)
    assert config.is_canonical_supercell(supercell) is False

    supercell = config.make_canonical_supercell(supercell)
    assert config.is_canonical_supercell(supercell) is True

    # check canonical supercell
    supercells.add(supercell)
    data = supercells.to_dict(version="1.0")
    assert "supercells" in data
    assert len(data["supercells"]) == 1
    assert "non_canonical_supercells" not in data
    supercells.remove(supercell)

    # check non-canonical supercell
    supercells.add(T)
    data = supercells.to_dict(version="1.0")
    assert "supercells" in data
    assert len(data["supercells"]) == 0
    assert "non_canonical_supercells" not in data
    supercells.remove(T)


def test_SupercellSet_from_dict_v1_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)

    data = {
        "supercells": {"SCEL2_1_2_1_1_0_0": [[-1, 0, 0], [0, 1, 1], [0, 1, -1]]},
        "version": "1.0",
    }
    supercells = config.SupercellSet.from_dict(data, prim)
    assert isinstance(supercells, config.SupercellSet)
    assert len(supercells) == 1


def test_SupercellRecord_repr(simple_cubic_binary_prim):
    import libcasm.xtal as xtal

    prim = config.Prim(simple_cubic_binary_prim)

    # Add supercells
    supercell_set = config.SupercellSet(prim)
    for superlat in xtal.enumerate_superlattices(
        unit_lattice=prim.xtal_prim.lattice(),
        point_group=prim.crystal_point_group.elements,
        max_volume=4,
    ):
        is_superlattice_of, T_float = superlat.is_superlattice_of(
            prim.xtal_prim.lattice()
        )
        supercell_set.add_by_transformation_matrix_to_super(
            np.rint(T_float).astype(int)
        )

    # Test print ConfigurationRecord
    import io
    from contextlib import redirect_stdout

    for record in supercell_set:
        assert isinstance(record, config.SupercellRecord)
        f = io.StringIO()
        with redirect_stdout(f):
            print(record)
        out = f.getvalue()
        assert "supercell" in out
        assert "supercell_name" in out
        assert "canonical_supercell_name" in out
        assert "is_canonical" in out


def test_SupercellSet_get_1(simple_cubic_binary_prim):
    prim = config.Prim(simple_cubic_binary_prim)
    supercells = config.SupercellSet(prim)

    T = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    supercell = config.make_canonical_supercell(config.Supercell(prim, T))
    record = supercells.add(supercell)
    supercell_name = record.supercell_name

    # get by supercell
    record_1 = supercells.get(supercell)
    assert record_1 is not None
    assert record_1 == record

    record_2 = supercells.get_supercell(supercell)
    assert record_2 == record_1

    # get by transformation matrix
    canonical_T = supercell.transformation_matrix_to_super
    record_3 = supercells.get(canonical_T)
    assert record_3 is not None
    assert record_3 == record

    record_4 = supercells.get_by_transformation_matrix_to_super(canonical_T)
    assert record_4 == record_3

    # get by record
    record_5 = supercells.get(record)
    assert record_5 is not None
    assert record_5 == record

    record_6 = supercells.get_record(record)
    assert record_6 == record_5

    # get by canonical name
    record_7 = supercells.get(supercell_name)
    assert record_7 is not None
    assert record_7 == record

    record_8 = supercells.get_by_canonical_name(supercell_name)
    assert record_8 == record_7

    # after removal, all get variants return None
    supercells.remove(supercell)

    assert supercells.get(supercell) is None
    assert supercells.get(canonical_T) is None
    assert supercells.get(record) is None
    assert supercells.get(supercell_name) is None
