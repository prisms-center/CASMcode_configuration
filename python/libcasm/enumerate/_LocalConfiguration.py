import copy
from functools import total_ordering
from typing import Optional

import libcasm.clusterography as casmclust
import libcasm.configuration as casmconfig
import libcasm.occ_events as occ_events
import libcasm.xtal as xtal

from ._OccEventPrimSymInfo import OccEventPrimSymInfo
from ._OccEventSupercellSymInfo import (
    OccEventSupercellSymInfo,
)


class OccEventSymInfo:
    """Shared information about an OccEvent used for transformations and conversions"""

    def __init__(
        self,
        event_prim_info: OccEventPrimSymInfo,
    ):
        """

        .. rubric:: Constructor

        Notes
        -----

        The :func:`OccEventSymInfo.init <libcasm.enumerate.OccEventSymInfo.init>`
        method is a more convenient way to construct an `OccEventSymInfo` object for
        most use cases.

        Parameters
        ----------
        event_prim_info: OccEventPrimSymInfo
            The information about the OccEvent with respect to the prim, which defines
            the meaning of the LocalConfiguration.pos attribute.

        """
        if not isinstance(event_prim_info, OccEventPrimSymInfo):
            raise ValueError(
                "Error in OccEventSymInfo constructor: "
                "event_prim_info must be an OccEventPrimSymInfo."
            )

        self.event_prim_info = event_prim_info
        """OccEventPrimSymInfo: Information about the OccEvent with respect to the prim,
        which defines the meaning of the LocalConfiguration.pos attribute."""

        self._all_event_supercell_info = dict()
        """dict[libcasm.configuration.Supercell, OccEventSupercellSymInfo]: Information 
        about the OccEvent with respect to the supercell, for each supercell."""

    def get_event_supercell_info(
        self,
        supercell: casmconfig.Supercell,
    ):
        """Get the OccEventSupercellSymInfo for a supercell.

        Parameters
        ----------
        supercell : libcasm.configuration.Supercell
            The supercell.

        Returns
        -------
        event_supercell_info : OccEventSupercellSymInfo
            The OccEventSupercellSymInfo for the supercell.
        """
        if supercell not in self._all_event_supercell_info:
            self._all_event_supercell_info[supercell] = OccEventSupercellSymInfo(
                event_prim_info=self.event_prim_info,
                supercell=supercell,
            )
        return self._all_event_supercell_info[supercell]

    def copy(self):
        return OccEventSymInfo(event_prim_info=self.event_prim_info.copy())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def to_data(self):
        """
        Serialize the `event_prim_info`.

        Returns
        -------
        prototype_event_data : dict
            The serialized prototype event data
        equivalents_info_data : dict
            The serialized equivalents info. Only the prototype and phenomenal
            clusters and equivalents generating op indices are included.
        """
        return self.event_prim_info.to_data()

    @staticmethod
    def init(
        prim: casmconfig.Prim,
        system: occ_events.OccSystem,
        prototype_event: occ_events.OccEvent,
        phenomenal_clusters: Optional[list[casmclust.Cluster]] = None,
        equivalent_generating_op_indices: Optional[list[int]] = None,
    ):
        """Construct OccEventSymInfo from the prim, system, and event

        Parameters
        ----------
        prim: libcasm.configuration.Prim
            The prim.
        system: libcasm.occ_events.OccSystem
            Occupation system, for index conversions.
        prototype_event: libcasm.occ_events.OccEvent
            The prototype event. The underlying cluster `prototype_event.cluster` must
            be chosen from one of the equivalents that is generated by
            :func:`~libcasm.clusterography.make_periodic_orbit` using the prim factor
            group.
        phenomenal_clusters: Optional[list[libcasm.clusterography.Cluster]]
            The clusters at which the equivalent events are located.
            By default, the phenomenal clusters are generated from `prototype_event` and
            the prim factor group. May be provided to ensure consistency of
            `(unitcell_index, equivalent_index)` event positions with those used by a
            local clexulator.

            Note that these clusters are expected to be with the
            site order as transformed from the prototype cluster by the equivalents
            generating factor group operations, without sites being sorted after
            transformation.

        equivalent_generating_op_indices: Optional[list[int]]
            The indices of prim factor group operations that generate the equivalent
            events from the prototype event, up to a translation. By default, the
            indices are generated from `event` and the prim factor group. May be
            provided to ensure consistency of `(unitcell_index, equivalent_index)` event
            positions with those used by a local clexulator.
        """
        return OccEventSymInfo(
            event_prim_info=OccEventPrimSymInfo(
                prim=prim,
                system=system,
                prototype_event=prototype_event,
                phenomenal_clusters=phenomenal_clusters,
                equivalent_generating_op_indices=equivalent_generating_op_indices,
            )
        )

    @staticmethod
    def from_data(
        prototype_event_data: dict,
        equivalents_info_data: dict,
        prim: casmconfig.Prim,
        system: occ_events.OccSystem,
    ):
        """
        Construct from the contents of "event.json" and "equivalents_info.json" files

        Parameters
        ----------
        prototype_event_data : dict
            The serialized prototype event data
        equivalents_info_data : dict
            The serialized equivalents info. The prototype and phenomenal clusters
            and equivalents generating op indices are needed; the local clusters are
            not needed.
        prim : libcasm.configuration.Prim
            The :class:`libcasm.configuration.Prim`
        system : libcasm.occ_events.OccSystem
            The :class:`libcasm.occ_events.OccSystem`

        Returns
        -------
        event_info : OccEventSymInfo
            The OccEventSymInfo object
        """
        return OccEventSymInfo(
            event_prim_info=OccEventPrimSymInfo.from_data(
                prototype_event_data=prototype_event_data,
                equivalents_info_data=equivalents_info_data,
                prim=prim,
                system=system,
            )
        )


@total_ordering
class LocalConfiguration:
    """An OccEvent-Configuration pair

    The position of an :class:`~libcasm.occ_events.OccEvent` with respect to a
    :class:`~libcasm.configuration.Configuration` is represented by a `pos` tuple
    with two integers, as `(unitcell_index, equivalent_index)`.

    The `equivalent_index` specifies the event orientation as the index into
    the orbit of equivalent events associated with the origin unit cell
    (:func:`libcasm.enumerate.OccEventPrimSymInfo.events`), and the
    `unitcell_index` specifies the unit cell in the supercell where the event
    is located (as the translation
    `UnitCellIndexConverter.unitcell(unitcell_index)`).

    Conversions between :class:`~libcasm.occ_events.OccEvent` and `pos` can be
    most easily done using for a particular supercell using
    :func:`OccEventSupercellSymInfo.coordinate
    <libcasm.enumerate.OccEventSupercellSymInfo.coordinate>`
    and
    :func:`OccEventSupercellSymInfo.event
    <libcasm.enumerate.OccEventSupercellSymInfo.event>`.

    The `equivalent_index` should be used as an index into
    :func:`libcasm.enumerate.OccEventPrimSymInfo.events`

    .. rubric:: Special Methods

    The multiplication operator ``X = lhs * rhs`` can be used to apply
    :class:`~libcasm.configuration.SupercellSymOp` to LocalConfiguration:

    - ``X=LocalConfiguration``, ``lhs=SupercellSymOp``, ``rhs=LocalConfiguration``:
      Copy and transform the event and configuration, returning the transformed
      LocalConfiguration, such that `X.configuration == lhs * rhs.configuration`, and
      `X.pos` is the position of the transformed event, standardized and within the
      supercell.


    Sort and compare LocalConfiguration:

    - LocalConfiguration can be sorted using ``<``, ``<=``, ``>``, ``>=``, and compared
      using ``==``, ``!=``:


    Additional methods:

    - LocalConfiguration may be copied with
      :func:`LocalConfiguration.copy <libcasm.enumerate.LocalConfiguration.copy>`,
      `copy.copy`, or `copy.deepcopy`.

    """

    def __init__(
        self,
        configuration: casmconfig.Configuration,
        pos: tuple[int, int],
        event_info: OccEventSymInfo,
    ):
        """

        .. rubric:: Constructor

        Parameters
        ----------
        configuration: libcasm.configuration.Configuration
            The configuration.
        pos: tuple[int, int]
            The position of the phenomenal cluster or event in the supercell, as
            `(unitcell_index, equivalent_index)`.
        event_info: OccEventSymInfo
            Information about the OccEvent with respect to the supercell, which
            defines the meaning of the `pos` attribute and can be shared for all
            LocalConfiguration with the same type of event.
        """

        self.configuration = configuration
        """libcasm.configuration.Configuration: The configuration."""

        self.pos = pos
        """tuple[int, int]: The position of the phenomenal cluster or event in the 
        supercell, as `(unitcell_index, equivalent_index)`."""

        self.event_info = event_info
        """OccEventSymInfo: Information about the OccEvent"""

        self._event_supercell_info = self.event_info.get_event_supercell_info(
            supercell=configuration.supercell
        )
        """OccEventSupercellSymInfo: Information about the OccEvent with respect to the 
        supercell, which defines the meaning of the `pos` attribute, which can be
        shared for all LocalConfiguration in the same supercell."""

    @staticmethod
    def from_event(
        configuration: casmconfig.Configuration,
        event: occ_events.OccEvent,
        event_info: OccEventSymInfo,
    ) -> "LocalConfiguration":
        """Convert a OccEvent and Configuration to a LocalConfiguration.

        Parameters
        ----------
        event : libcasm.occ_events.OccEvent
            The OccEvent.
        configuration : libcasm.configuration.Configuration
            The Configuration.
        event_info: OccEventSymInfo
            Information about the OccEvent with respect to the supercell, which
            defines the meaning of the `pos` attribute and can be shared for all
            LocalConfiguration with the same type of event.

        Returns
        -------
        local_configuration : LocalConfiguration
            The equivalent LocalConfiguration.

        """
        supercell = configuration.supercell
        event_supercell_info = event_info.get_event_supercell_info(supercell)
        pos = event_supercell_info.coordinate(event)
        return LocalConfiguration(
            pos=pos,
            configuration=configuration,
            event_info=event_info,
        )

    @property
    def event(self):
        """libcasm.occ_events.OccEvent: The OccEvent associated with the
        LocalConfiguration, as determined from `pos`."""
        return self._event_supercell_info.event(self.pos)

    def copy(self):
        """Return a copy of the LocalConfiguration.

        Returns
        -------
        new_local_config: LocalConfiguration
            A copy of the LocalConfiguration. The `configuration` and `pos` attributes
            are copied and the `event_supercell_info` attribute is shared.
        """
        return LocalConfiguration(
            pos=copy.deepcopy(self.pos),
            configuration=self.configuration.copy(),
            event_info=self.event_info,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def __rmul__(
        self,
        lhs: casmconfig.SupercellSymOp,
    ):
        if not isinstance(lhs, casmconfig.SupercellSymOp):
            raise ValueError(
                "Error in LocalConfiguration.__rmul__: lhs must be a SupercellSymOp"
            )
        if lhs.supercell != self.configuration.supercell:
            raise ValueError(
                "Error in LocalConfiguration.__rmul__: lhs must be a SupercellSymOp "
                "for the same supercell as the LocalConfiguration."
            )

        # Apply op to event and get final_pos
        curr = self._event_supercell_info
        final_event = curr.copy_apply_supercell_symop(
            op=lhs,
            occ_event=self.event,
        )
        final_pos = curr.coordinate(final_event)

        return LocalConfiguration(
            pos=final_pos,
            configuration=lhs * self.configuration,
            event_info=self.event_info,
        )

    def __eq__(self, other):
        if self.event_info is not other.event_info:
            raise ValueError(
                "LocalConfiguration objects must have the same event_info "
                "to be compared."
            )
        return self.pos == other.pos and self.configuration == other.configuration

    def __lt__(self, other):
        if self.event_info is not other.event_info:
            raise ValueError(
                "LocalConfiguration objects must have the same event_info "
                "to be compared."
            )
        # Compare pos, then configuration
        if self.pos < other.pos:
            return True
        if self.pos > other.pos:
            return False
        return self.configuration < other.configuration

    def __repr__(self):
        return xtal.pretty_json(self.to_dict())

    def to_dict(
        self,
        write_prim_basis: bool = False,
    ):
        """Represent the LocalConfiguration as a Python dict

        Note
        ----
        The `event_supercell_info` is not included to avoid excessive duplication.

        Parameters
        ----------
        write_prim_basis: bool = False
            If True, write DoF values using the prim basis. Default (False) is to
            write DoF values in the standard basis.

        Returns
        -------
        data: dict
            A Python dict representation of the LocalConfiguration.
        """
        return {
            "configuration": self.configuration.to_dict(
                write_prim_basis=write_prim_basis
            ),
            "pos": [self.pos[0], self.pos[1]],
        }

    @staticmethod
    def from_dict(
        data: dict,
        supercells: casmconfig.SupercellSet,
        event_info: OccEventSymInfo,
    ):
        """Construct the LocalConfiguration from a Python dict

        Parameters
        ----------
        data : dict
            A :class:`~libcasm.enumerate.LocalConfiguration` as a dict.
        supercells : libcasm.configuration.SupercellSet
            A :class:`~libcasm.configuration.SupercellSet`, which holds shared
            supercells in order to avoid duplicates.
        event_info: OccEventSymInfo
            Information about the OccEvent with respect to the supercell, which
            defines the meaning of the `pos` attribute and can be shared for all
            LocalConfiguration with the same type of event.

        Returns
        -------
        local_configuration : libcasm.enumerate.LocalConfiguration
            The :class:`~libcasm.configuration.Configuration` constructed from the dict.
            The constructed LocalConfiguration is not automatically added to the
            LocalConfigurationList.
        """
        return LocalConfiguration(
            configuration=casmconfig.Configuration.from_dict(
                data=data["configuration"],
                supercells=supercells,
            ),
            pos=(data["pos"][0], data["pos"][1]),
            event_info=event_info,
        )


class LocalConfigurationList:
    """A list of LocalConfigurations with the same type of OccEvent.

    All the :class:`~libcasm.enumerate.LocalConfiguration` objects in the list must have
    :class:`~libcasm.occ_events.OccEvent` that are equivalent by prim factor group
    symmetry and share the same :class:`libcasm.enumerate.OccEventSymInfo`.

    .. rubric:: Special methods

    - `local_config in local_config_list`: Check if a LocalConfiguration is in the
      LocalConfigurationList.
    - `len(local_config_list)`: Get the number of LocalConfigurations
    - `local_config = local_config_list[i]`: Get the i-th LocalConfiguration
    - `local_config_list[i] = local_config`: Set the i-th LocalConfiguration
    - `del local_config_list[i]`: Delete the i-th LocalConfiguration
    - `for lc in local_config_list`: Iterate over the LocalConfigurations

    """

    def __init__(
        self,
        event_info: OccEventSymInfo,
        local_configurations: Optional[list[LocalConfiguration]] = None,
    ):
        """

        .. rubric:: Constructor

        Parameters
        ----------
        event_info: OccEventSymInfo
            Information about the OccEvent with respect to the supercell, which
            defines the meaning of the `pos` attribute and can be shared for all
            LocalConfiguration with the same type of event.
        local_configurations : Optional[list[LocalConfiguration]] = None
            An optional list of LocalConfiguration to initialize
            LocalConfigurationList with. Does not create a copy.
        """

        self.event_info = event_info
        """OccEventSymInfo: Information about the OccEvent, which defines the
        meaning of the LocalConfiguration.pos attribute and can be shared for all
        LocalConfiguration with the same type of event."""

        if local_configurations is None:
            local_configurations = list()

        try:
            for x in local_configurations:
                self._check_value(x)
        except ValueError:
            raise ValueError(
                "Error: local_configurations must be a list of LocalConfiguration "
                "which share the same event_info."
            )

        self._local_configurations = local_configurations
        """list[LocalConfiguration]: The list of local configurations."""

    def _check_value(self, value):
        """Check insertion value

        - Raise if value is not a LocalConfiguration
        - Raise if the event_info is not shared
        """

        if not isinstance(value, LocalConfiguration):
            raise ValueError("Error: value must be a LocalConfiguration.")
        if value.event_info is not self.event_info:
            raise ValueError(
                "Error: LocalConfiguration must have the same event_info as the "
                "LocalConfigurationList."
            )
        return value

    def as_local_configuration(
        self,
        event: occ_events.OccEvent,
        configuration: casmconfig.Configuration,
    ) -> LocalConfiguration:
        """Convert a OccEvent and Configuration to a LocalConfiguration.

        Parameters
        ----------
        event : libcasm.occ_events.OccEvent
            The OccEvent.
        configuration : libcasm.configuration.Configuration
            The Configuration.

        Returns
        -------
        local_configuration : LocalConfiguration
            The equivalent LocalConfiguration.

        """
        supercell = configuration.supercell
        event_supercell_info = self.event_info.get_event_supercell_info(supercell)
        pos = event_supercell_info.coordinate(event)
        return LocalConfiguration(
            pos=pos,
            configuration=configuration,
            event_info=self.event_info,
        )

    def __contains__(self, value):
        value = self._check_value(value)
        return value in self._local_configurations

    def __len__(self):
        return len(self._local_configurations)

    def __getitem__(self, i):
        return self._local_configurations[i]

    def __setitem__(self, i, value):
        value = self._check_value(value)
        self._local_configurations[i] = value

    def __delitem__(self, i):
        del self._local_configurations[i]

    def __iter__(self):
        return iter(self._local_configurations)

    def append(self, value):
        """Append a LocalConfiguration to the list.

        Parameters
        ----------
        value : libcasm.enumerate.LocalConfiguration
            The value to append to the list.
        """

        value = self._check_value(value)
        self._local_configurations.append(value)

    def sort(self):
        """Sort the list of LocalConfigurations in place."""
        self._local_configurations.sort()

    def index(self, value):
        """Find the index of an equivalent LocalConfiguration in the list.

        Parameters
        ----------
        value : libcasm.enumerate.LocalConfiguration
            The value to find in the list.

        Returns
        -------
        index : int
            The zero-based index in the list of the first item whose value is equal to
            `value`. Raises a ValueError if there is no such item.
        """

        value = self._check_value(value)
        return self._local_configurations.index(value)

    def clear(self):
        """Remove all LocalConfigurations from the list."""
        self._local_configurations.clear()

    def copy(self):
        """Return a copy of the LocalConfigurationList.

        Returns
        -------
        new_list: LocalConfigurationList
            A copy of the LocalConfigurationList, where all the LocalConfigurations are
            copied, and the `event_info` is shared with the original.
        """
        return LocalConfigurationList(
            event_info=self.event_info,
            local_configurations=[lc.copy() for lc in self._local_configurations],
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def to_dict(self):
        """Represent the LocalConfigurationList as a Python dict

        Returns
        -------
        data : dict
            A Python dict representation of the LocalConfigurationList.
        """
        (
            prototype_event_data,
            equivalents_info_data,
        ) = self.event_info.to_data()
        return {
            "prototype_event": prototype_event_data,
            "equivalents_info": equivalents_info_data,
            "local_configurations": [lc.to_dict() for lc in self._local_configurations],
        }

    @staticmethod
    def from_dict(
        data: dict,
        prim: casmconfig.Prim,
        system: occ_events.OccSystem,
        supercells: casmconfig.SupercellSet,
    ):
        """Construct a LocalConfigurationList from a Python dict

        Parameters
        ----------
        data : dict
            A Python dict representation of the LocalConfigurationList.
        prim : libcasm.configuration.Prim
            The :class:`libcasm.configuration.Prim`
        system : libcasm.occ_events.OccSystem
            The :class:`libcasm.occ_events.OccSystem`
        supercells : libcasm.configuration.SupercellSet
            A :class:`~libcasm.configuration.SupercellSet`, which holds shared
            supercells in order to avoid duplicates.

        Returns
        -------
        local_config_list : LocalConfigurationList
            The LocalConfigurationList.
        """
        event_info = OccEventSymInfo.from_data(
            prototype_event_data=data["prototype_event"],
            equivalents_info_data=data["equivalents_info"],
            prim=prim,
            system=system,
        )
        local_configurations = [
            LocalConfiguration.from_dict(
                data=lc,
                supercells=supercells,
                event_info=event_info,
            )
            for lc in data["local_configurations"]
        ]
        return LocalConfigurationList(
            event_info=event_info,
            local_configurations=local_configurations,
        )


def make_canonical_local_configuration(
    initial: LocalConfiguration,
    in_canonical_pos: bool = True,
    in_canonical_supercell: bool = False,
    apply_event_occupation: bool = True,
):
    """Make the canonical form of a local configuration

    Notes
    -----

    Depending on the use case, a LocalConfiguration may be made canonical with
    respect to:

    1. The event invariant group (transform the configuration, but leave the
       event position and supercell unchanged) using `in_canonical_pos=False` and
       `in_canonical_supercell=False`, or
    2. The supercell invariant group (transform the event position and the
       configuration, but leave supercell the same) using `in_canonical_pos=True`
       and `in_canonical_supercell=False`, or
    3. The prim factor group (transform the supercell, the event position, and the
       configuration) `in_canonical_pos=True` and `in_canonical_supercell=True`.

    The default for this method is make a LocalConfiguration canonical with respect
    to the supercell invariant group, i.e., `in_canonical_pos=True` and
    `in_canonical_supercell=False`.

    Additionally, the occupation of the event can be applied to the configuration
    or not using `apply_event_occupation`. Using `apply_event_occupation=False` is
    useful for use applications such as generating unique positions in a primitive
    configuration, while `apply_event_occupation=True` is useful for applications
    such as generating unique local configurations in a larger supercell for
    calculations.

    Parameters
    ----------
    initial: libcasm.enumerate.LocalConfiguration]
        The initial LocalConfiguration to transform.
    in_canonical_pos: bool = True
        If True, transform `initial` to put `initial.pos` in the
        canonical position in the supercell. Else, keep `initial.pos` in its current
        position and only transform `initial.configuration`.
    in_canonical_supercell: bool = False
        If True, first transform `initial` to put both `pos` and `configuration`
        into the canonical equivalent supercell. If True, `in_canonical_pos` must
        also be True.
    apply_event_occupation: bool = True
        If True, apply the occupation of the event to the configuration. If False,
        maintain the current configuration occupation.

    Returns
    -------
    final: libcasm.enumerate.LocalConfiguration
        The final LocalConfiguration after the transformation.
    """
    curr = initial._event_supercell_info
    (
        _final_config,
        _final_pos,
        _final_event_supercell_info,
    ) = curr.make_canonical_local_configuration(
        configuration=initial.configuration,
        pos=initial.pos,
        in_canonical_pos=in_canonical_pos,
        in_canonical_supercell=in_canonical_supercell,
        apply_event_occupation=apply_event_occupation,
    )

    return LocalConfiguration(
        pos=_final_pos,
        configuration=_final_config,
        event_info=initial.event_info,
    )