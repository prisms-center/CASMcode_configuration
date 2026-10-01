#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <utility>

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/configuration/group/Group.hh"
#include "casm/configuration/group/subgroups.hh"
#include "casm/configuration/sym_info/factor_group.hh"
#include "casm/configuration/sym_info/io/json/SymGroup_json_io.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/crystallography/io/SymInfo_json_io.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"
#include "casm/global/pybind11_helpers.hh"
#include "casm/global/threads.hh"
#include "pybind11_json/pybind11_json.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// SymGroup

std::shared_ptr<sym_info::SymGroup> make_symgroup(
    std::vector<xtal::SymOp> const &element,
    group::MultiplicationTable const &multiplication_table) {
  return std::make_shared<sym_info::SymGroup>(element, multiplication_table);
}

std::shared_ptr<sym_info::SymGroup const> make_symgroup_subgroup(
    std::shared_ptr<sym_info::SymGroup const> const &group,
    std::set<Index> const &head_group_index,
    std::optional<std::vector<xtal::SymOp>> element = std::nullopt) {
  std::shared_ptr<sym_info::SymGroup const> head_group;
  if (!group->head_group) {
    head_group = group;
  } else {
    head_group = group->head_group;
  }
  if (!element.has_value()) {
    return std::make_shared<sym_info::SymGroup>(head_group, head_group_index);
  }
  return std::make_shared<sym_info::SymGroup>(head_group, *element,
                                              head_group_index);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_sym_info, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Crystallographic symmetry information

        libcasm.sym_info
        ----------------

        The libcasm.sym_info package constructs crystallographic symmetry information.

    )pbdoc";
  py::module::import("libcasm.xtal");
  py::module::import("libcasm.group");

  py::class_<sym_info::SymGroup, group::GenericGroup,
             std::shared_ptr<sym_info::SymGroup>>(m, "SymGroup", R"pbdoc(
      Data structure holding group elements and other group info, such as
      group-subgroup relationships.

      The :class:`~libcasm.sym_info.SymGroup` class goes beyond storing a ``list[SymOp]``
      to include the multiplication table and inverse element table. The SymGroup class
      may represent a head group, or a subgroup.

      When representing a subgroup, the head group can be obtained as a shared pointer
      from :func:`~libcasm.sym_info.SymGroup.head_group`. Pointers to subgroups are not
      stored by the head group. For subgroups, there is a list of indices indicating which
      element in the head group each subgroup element corresponds to
      (:func:`~libcasm.sym_info.SymGroup.head_group_index`).

      The :class:`libcasm.sym_info` package provides factory functions for the common use
      cases of constructing the prim factor group and prim point group:

      .. rubric:: Factory Functions

      - :func:`~libcasm.sym_info.make_factor_group`:  Make the \
        :class:`libcasm.xtal.Prim` factor group
      - :func:`~libcasm.sym_info.make_point_group`: Make the \
        :class:`libcasm.xtal.Prim` point group
      - :func:`~libcasm.sym_info.make_lattice_point_group`: Make the \
        :class:`libcasm.xtal.Lattice` point group

      .. rubric:: Special methods

      - Get the `i`-th element of the group using: ``element = group[i]``

      )pbdoc")
      .def(py::init(&make_symgroup), R"pbdoc(

          .. rubric:: Constructor

          Notes
          -----
          It is recommended to use the factory functions provided by CASM
          to construct SymGroup. The factory functions sort elements by class
          and type, resulting in a standard ordering of elements.

          Parameters
          ----------
          elements: list[libcasm.xtal.SymOp]
              The symmetry operations forming the group.
          multiplication_table: list[list[int]]
              The multiplication table element
              ``multiplication_table[i][j] == k`` represents that
              ``elements[k]`` is equivalent to ``elements[i] * elements[j]``.
          )pbdoc",
           py::arg("elements"), py::arg("multiplication_table"))
      .def_static(
          "from_elements",
          [](std::vector<xtal::SymOp> elements, xtal::Lattice const &lattice,
             bool sort) -> std::shared_ptr<sym_info::SymGroup const> {
            if (sort) {
              return sym_info::make_symgroup(elements, lattice);
            } else {
              return sym_info::make_symgroup_without_sorting(elements, lattice);
            }
          },
          py::arg("elements"), py::arg("lattice"), py::arg("sort") = true)
      .def("make_subgroup", &make_symgroup_subgroup, R"pbdoc(
          Make a subgroup

          Parameters
          ----------
          head_group_index: set[int]
              Indices of elements in the head group (which may or may not be
              `self`) to include in the subgroup.
          elements: Optional[list[libcasm.xtal.SymOp]] = None
              If not None, use the provided elements for subgroup elements list.
              This allows representing subgroups of the space group with
              reference to the factor group. For example, cluster invariant
              groups need SymOp that have the correct translation to leave the
              cluster invariant, which may have a different translation than
              the corresponding element in the factor group elements list.
          )pbdoc",
           py::arg("head_group_index"), py::arg("element") = std::nullopt)
      .def_property_readonly(
          "size",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->size();
          },
          R"pbdoc(
          int: The number of elements in the group.
          )pbdoc")
      .def_property_readonly(
          "elements",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->element;
          },
          R"pbdoc(
          list[libcasm.xtal.SymOp]: The elements list
          )pbdoc")
      .def(
          "__getitem__",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             Index i) { return symgroup->element.at(i); },
          py::arg("i"), R"pbdoc(
          libcasm.xtal.SymOp: The `i`-th element in the group.
          )pbdoc")
      .def("__len__",
           [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
             return symgroup->element.size();
           })
      .def(
          "__iter__",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return py::make_iterator(symgroup->element.begin(),
                                     symgroup->element.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def_property_readonly(
          "multiplication_table",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->multiplication_table;
          },
          R"pbdoc(
          list[list[int]]: The multiplication table.

          The multiplication table element `multiplication_table[i][j] == k`
          represents that
          ``np.allclose(elements[k], elements[i] @ elements[j]) == True``.
          )pbdoc")
      .def(
          "conjugacy_classes",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            std::vector<std::vector<Index>> conjugacy_classes;
            for (Index i = 0; i < symgroup->element.size(); ++i) {
              Index cc = symgroup->class_of(i);
              if (cc >= conjugacy_classes.size()) {
                conjugacy_classes.resize(cc + 1);
              }
              conjugacy_classes[cc].push_back(i);
            }
            return conjugacy_classes;
          },
          R"pbdoc(
          Returns the conjugacy classes

          Returns
          -------
          conjugacy_classes: list[list[int]]
              ``conjugacy_classes[i]`` is a list of the indices of elements in
              the `i`-th class."
          )pbdoc")
      .def(
          "mult",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup, Index i,
             Index j) { return symgroup->mult(i, j); },
          py::arg("i"), py::arg("j"),
          R"pbdoc(
          Returns the index of the element product.

          Parameters
          ----------
          i: int
              lhs element index.
          j: int
              rhs element index.

          Returns
          -------
          k: int
              The index ``k == multiplication_table[i][j]``.
          )pbdoc")
      .def_property_readonly(
          "is_subgroup",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->head_group != nullptr;
          },
          R"pbdoc(
          bool: True if this is a subgroup, False otherwise.
          )pbdoc")
      .def_property_readonly(
          "head_group",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            std::optional<std::shared_ptr<sym_info::SymGroup const>> head_group;
            if (symgroup->head_group != nullptr) {
              head_group = symgroup->head_group;
            }
            return head_group;
          },
          R"pbdoc(
          Optional[SymGroup]: The head group if this is a subgroup, else None.
          )pbdoc")
      .def_property_readonly(
          "head_group_index",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->head_group_index;
          },
          R"pbdoc(
          list[int]: The list of head group indices (guaranteed sorted)

          If this is the head group, then
          ``group.head_group_index == [0, 1, 2, ...]``.

          If this is a sub group, then ``subgroup.element[i]`` is the same element
          as ``subgroup.head_group.element[subgroup.head_group_index[i]]``.

          )pbdoc")
      .def_property_readonly(
          "inverse_index",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup) {
            return symgroup->inverse_index;
          },
          R"pbdoc(
          list[int]: The list of inverse indices

          Represents that ``group.element[group.inverse_index[i]]`` is the
          inverse of ``group.element[i]``.
          )pbdoc")
      .def(
          "inv",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             Index i) { return symgroup->inv(i); },
          py::arg("i"), R"pbdoc(
          Returns the index of the inverse of an element

          Parameters
          ----------
          i: int
              The element index.

          Returns
          -------
          i_inverse: int
              The index the inverse of the `i`-th element.
          )pbdoc")
      .def(
          "class_of",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             Index i) { return symgroup->class_index[i]; },
          py::arg("i"), R"pbdoc(
          Returns the index of the conjugacy class containing an element

          Parameters
          ----------
          i: int
              The element index.

          Returns
          -------
          i_class: int
              The index the conjugacy class containing the `i`-th element.
          )pbdoc")
      .def(
          "brief_cart",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             xtal::Lattice const &lattice, int index_from) {
            std::stringstream ss;
            int index = index_from;
            for (auto const &op : symgroup->element) {
              xtal::SymInfo syminfo(op, lattice);
              ss << index << ": "
                 << to_brief_unicode(syminfo, xtal::SymInfoOptions(CART))
                 << std::endl;
              index += 1;
            }
            return ss.str();
          },
          py::arg("lattice"), py::arg("index_from") = 1, R"pbdoc(
          Brief description, in Cartesian coordinates

          Parameters
          ----------
          lattice: libcasm.xtal.Lattice
              The lattice.
          index_from: int = 1
              The first element index.

          Returns
          -------
          brief_desc_cart: str
              Brief description of the group elements, in Cartesian coordinates.
          )pbdoc")
      .def(
          "brief_frac",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             xtal::Lattice const &lattice, int index_from) {
            std::stringstream ss;

            int index = index_from;
            for (auto const &op : symgroup->element) {
              xtal::SymInfo syminfo(op, lattice);
              ss << index << ": "
                 << to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC))
                 << std::endl;
              index += 1;
            }
            return ss.str();
          },
          py::arg("lattice"), py::arg("index_from") = 1, R"pbdoc(
          Brief description, in fractional coordinates

          Parameters
          ----------
          lattice: libcasm.xtal.Lattice
              The lattice.
          index_from: int = 1
              The first element index.

          Returns
          -------
          brief_desc_cart: str
              Brief description of the group elements, in fractional
              coordinates.
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data, xtal::Lattice const &lattice) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;
            jsonParser json{data};
            InputParser<std::shared_ptr<sym_info::SymGroup const>> parser(
                json, lattice);
            std::runtime_error error_if_invalid{
                "Error in libcasm.sym_info.SymGroup.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          R"pbdoc(
          Construct a SymGroup from a Python dict.

          The
          `SymGroup reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/symmetry/SymGroup/>`_
          documents the expected format.

          Parameters
          ----------
          lattice : libcasm.xtal.Lattice
              The lattice, used to construct the multiplication table.

          Returns
          -------
          symgroup: libcasm.sym_info.SymGroup
              The SymGroup
          )pbdoc",
          py::arg("data"), py::arg("lattice"))
      .def(
          "to_dict",
          [](std::shared_ptr<sym_info::SymGroup const> const &symgroup,
             xtal::Lattice const &lattice) {
            jsonParser json;
            to_json(symgroup, json, lattice);
            return static_cast<nlohmann::json>(json);
          },
          py::arg("lattice"),
          R"pbdoc(
          Represent the SymGroup as a Python dict

          Parameters
          ----------
          lattice : libcasm.xtal.Lattice
              The lattice, used to construct the fractional coordinate
              representation and symmetry operation info.

          Returns
          -------
          data : dict
              The
              `SymGroup reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/symmetry/SymGroup/>`_
              documents the expected format.

          )pbdoc");

  m.def("make_lattice_point_group", &sym_info::make_lattice_point_group,
        R"pbdoc(
        Construct the lattice point group as a SymGroup

        Parameters
        ----------
        lattice: libcasm.xtal.Lattice
            The :class:`libcasm.xtal.Lattice`.

        Returns
        -------
        lattice_point_group : SymGroup
            The group which leaves `lattice` invariant
        )pbdoc",
        py::arg("lattice"));

  m.def("make_factor_group", &sym_info::make_factor_group, R"pbdoc(
        Construct the prim factor group as a SymGroup

        Parameters
        ----------
        xtal_prim: libcasm.xtal.Prim
            The :class:`libcasm.xtal.Prim`.

        Returns
        -------
        factor_group : SymGroup
            The group which leaves `xtal_prim` invariant
        )pbdoc",
        py::arg("xtal_prim"));

  m.def("make_point_group", &sym_info::make_point_group, R"pbdoc(
        Construct the prim point group as a SymGroup

        Parameters
        ----------
        xtal_prim: libcasm.xtal.Prim
            The :class:`libcasm.xtal.Prim`.

        factor_group: SymGroup
            The factor group of `prim`, as calculated by
            :func:`~libcasm.sym_info.make_factor_group`.

        Returns
        -------
        point_group : SymGroup
            The prim point group operations, constructed by removing the translation from
            prim factor group operations. Degenerate symmetry operations are not added. The
            resulting point_group is its own head group, not a subgroup of the input
            factor_group.

        )pbdoc",
        py::arg("xtal_prim"), py::arg("factor_group"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
