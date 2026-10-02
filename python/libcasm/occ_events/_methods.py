import json
import pathlib

import libcasm.clusterography as clust
import libcasm.occ_events._occ_events as _occ_events
import libcasm.sym_info as sym_info
import libcasm.xtal as xtal


def save_occevent(
    root: pathlib.Path,
    name: str,
    occ_event: _occ_events.OccEvent,
    system: _occ_events.OccSystem,
):
    """Save an OccEvent

    Saves an :class:`~libcasm.occ_events.OccEvent` to:

    - root / "events" / "event.<name>" / "event.json"

    Parameters
    ----------
    root: pathlib.Path
        Path to the parent of the "events" directory.
    name: str
        A name for the event, such as "1NN_A_Va".
    occ_event: ~libcasm.occ_events.OccEvent
        The event to save
    system: ~libcasm.occ_events.OccSystem
        An :class:`~libcasm.occ_events.OccSystem`, used for indexing

    """
    root = pathlib.Path(root)
    path = root / "events" / ("event." + name) / "event.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        data = occ_event.to_dict(system)
        f.write(xtal.pretty_json(data))


def load_occevent(
    root: pathlib.Path, name: str, system: _occ_events.OccSystem
) -> _occ_events.OccEvent:
    """Load a saved OccEvent

    Loads an :class:`~libcasm.occ_events.OccEvent` from:

    - root / "events" / "event.<name>" / "event.json"

    Parameters
    ----------
    root: pathlib.Path
        Path to the parent of the "events" directory.
    name: str
        A unique name for the event, such as "1NN_A_Va".
    system: ~libcasm.occ_events.OccSystem
        An :class:`~libcasm.occ_events.OccSystem`, used for indexing

    Returns
    -------
    occ_event: ~libcasm.occ_events.OccEvent
        The saved :class:`~libcasm.occ_events.OccEvent`.
    """
    root = pathlib.Path(root)
    path = root / "events" / ("event." + name) / "event.json"
    with open(path, "r") as f:
        data = json.load(f)
        occ_event = _occ_events.OccEvent.from_dict(data, system)
    return occ_event


def make_canonical_occevent(xtal_prim: xtal.Prim, occ_event: _occ_events.OccEvent):
    """Construct the canonical equivalent OccEvent

    Generates an orbit of :class:`~libcasm.occ_events.OccEvent` and
    returns the "greatest" element of the orbit.

    Parameters
    ----------
    xtal_prim: libcasm.xtal.Prim
        The Prim structure
    occ_event: ~libcasm.occ_events.OccEvent
        An :class:`~libcasm.occ_events.OccEvent`.

    Returns
    -------
    canonical_occ_event: ~libcasm.occ_events.OccEvent
        The canonical equivalent :class:`~libcasm.occ_events.OccEvent`.
    """
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    occevent_symgroup_rep = _occ_events.make_occevent_symgroup_rep(
        prim_factor_group.elements, xtal_prim
    )
    occevent_orbit = _occ_events.make_prim_periodic_orbit(
        occ_event, occevent_symgroup_rep
    )
    return occevent_orbit[-1]


def make_occevent_cluster_specs(
    xtal_prim: xtal.Prim,
    phenomenal_occ_event: _occ_events.OccEvent,
    max_length: list[float],
    cutoff_radius: list[float],
    custom_generators: list[clust.ClusterOrbitGenerator] = [],
) -> clust.ClusterSpecs:
    """Construct ClusterSpecs for local-cluster orbits around an OccEvent

    Parameters
    ----------
    xtal_prim: libcasm.xtal.Prim
        The Prim structure
    phenomenal_occ_event: ~libcasm.occ_events.OccEvent
        The orbit generating group is the subgroup of the prim factor
        group that leaves `phenomenal_occ_event` invariant.
    max_length: list[float]
        The maximum site-to-site distance to allow in clusters, by number
        of sites in the cluster. Example: `[0.0, 0.0, 5.0, 4.0]` specifies
        that pair clusters up to distance 5.0 and triplet clusters up to
        distance 4.0 should be included. The null cluster and point
        cluster values (elements 0 and 1) are arbitrary.
    cutoff_radius: list[float]
        For local clusters, the maximum distance of sites from any
        phenomenal cluster site to include in the local environment, by
        number of sites in the cluster. The null cluster value
        (element 0) is arbitrary.
    custom_generators: list[~libcasm.clusterography.ClusterOrbitGenerator]=[]
          Specifies clusters that should be uses to construct orbits
          regardless of the max_length or cutoff_radius parameters

    Returns
    -------
    cluster_specs: ~libcasm.clusterography.ClusterSpecs
        The resulting ClusterSpecs
    """
    prim_factor_group = sym_info.make_factor_group(xtal_prim)
    symgroup_rep = _occ_events.make_occevent_symgroup_rep(
        prim_factor_group.elements, xtal_prim
    )
    occevent_group = _occ_events.make_occevent_group(
        occ_event=phenomenal_occ_event,
        group=prim_factor_group,
        lattice=xtal_prim.lattice(),
        occevent_symgroup_rep=symgroup_rep,
    )
    return clust.ClusterSpecs(
        xtal_prim=xtal_prim,
        generating_group=occevent_group,
        max_length=max_length,
        phenomenal=phenomenal_occ_event.cluster(),
        cutoff_radius=cutoff_radius,
        custom_generators=custom_generators,
    )


def make_canonical_prim_periodic_occevents(
    system: _occ_events.OccSystem,
    cluster_specs: clust.ClusterSpecs,
    occevent_counter_params: dict = {},
    custom_events: list[_occ_events.OccEvent] = [],
) -> list[_occ_events.OccEvent]:
    """Enumerate symmetrically distinct OccEvent

    Parameters
    ----------
    system: ~libcasm.occ_events.OccSystem
        An :class:`~libcasm.occ_events.OccSystem`, used for indexing
    cluster_specs: ~libcasm.clusterography.ClusterSpecs
        ClusterSpecs specifying local-cluster orbits on which to construct
        OccEvent
    occevent_counter_params: dict
        Options filtering which OccEvent are generated. Includes:

        Filter by cluster size:

        - "min_cluster_size": Optional[int] = None
        - "max_cluster_size": Optional[int] = None
        - "required_cluster_size": Optional[int] = None

        Filter by sublattice, using sublattice indices:

        - "excluded_sublattices": Optional[list[int]] = None, Do not include
          events that contain any of these sublattices.
        - "required_sublattices": Optional[list[int]] = None, Only include
          events that contain these sublattices. Events may contain additional
          sublattices not specified in `required_sublattices`.
        - "n_required_sublattices": Optional[int] = None, Set how many
          sublattices from `required_sublattices` the event must contain.
          The default is that the event must contain every sublattice in
          `required_sublattices`.

        Filter by occupation, using occupation variable on each cluster site:

        - "required_occ_init": Optional[list[int]] = None
        - "required_occ_final": Optional[list[int]] = None

        Filter by atom type count, using indices defined by the order in
        :py:class:`~libcasm.occ_events.OccSystem.atom_name_list`:

        - "min_init_atom_count": Optional[list[int]] = None
        - "max_init_atom_count": Optional[list[int]] = None
        - "required_init_atom_count": Optional[list[int]] = None
        - "min_final_atom_count": Optional[list[int]] = None
        - "max_final_atom_count": Optional[list[int]] = None
        - "required_final_atom_count": Optional[list[int]] = None

        Filter by molecule type count, using indices defined by the order in
        :py:class:`~libcasm.occ_events.OccSystem.chemical_name_list`:

        - "min_init_molecule_count": Optional[list[int]] = None
        - "max_init_molecule_count": Optional[list[int]] = None
        - "required_init_molecule_count": Optional[list[int]] = None
        - "min_final_molecule_count": Optional[list[int]] = None
        - "max_final_molecule_count": Optional[list[int]] = None
        - "required_final_molecule_count": Optional[list[int]] = None

        Filter by molecule orientation type count, using indices defined by the order
        in :py:class:`~libcasm.occ_events.OccSystem.orientation_name_list`:

        - "min_init_orientation_count": Optional[list[int]] = None
        - "max_init_orientation_count": Optional[list[int]] = None
        - "required_init_orientation_count": Optional[list[int]] = None
        - "min_final_orientation_count": Optional[list[int]] = None
        - "max_final_orientation_count": Optional[list[int]] = None
        - "required_final_orientation_count": Optional[list[int]] = None

        Filter by type of event:

        - "allow_subcluster_events": Optional[bool] = False, Optionally include
          multisite events that do not result in any change on some sites.
        - "do_not_allow_breakup": Optional[bool] = False, Do not include events that
          break up a multi-atom molecule.
        - "skip_direct_exchange": Optional[bool] = True, Optionally include events in
          which non-vacancies directly exchange sites.
        - "print_state_info": Optional[bool] = False, Print information about the
          step-by-step state of the algorithm.

    custom_events: list[~libcasm.clusterography.ClusterOrbitGenerator]=[]
          Specifies OccEvent that should be included in the results
          regardless of the other options.

    Returns
    -------
    canonical_prim_periodic_occevents: list[~libcasm.clusterography.OccEvent]
        The resulting OccEvent
    """
    return _occ_events.make_canonical_prim_periodic_occevents(
        system, cluster_specs, occevent_counter_params, custom_events
    )
