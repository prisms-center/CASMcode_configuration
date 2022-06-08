#include "casm/configuration/canonical_form.hh"

#include "casm/crystallography/CanonicalForm.hh"

namespace CASM {
namespace config {

/// \brief Return true if supercell lattice is in canonical form
bool is_canonical(Supercell const &supercell) {
  return xtal::canonical::check(supercell.superlattice.superlattice(),
                                supercell.prim->sym_info.point_group->element);
}

/// \brief Return a shared supercell that compares greater to all equivalents
///     with respect to prim point group symmetry
///
/// The result, `canonical_supercell` satisfies for all `op` in
/// `supercell.prim->sym_info.point_group->element`:
///     canonical_supercell->superlattice.superlattice() >=
///         sym::copy_apply(op, supercell.superlattice.superlattice())
std::shared_ptr<Supercell const> make_canonical_form(
    Supercell const &supercell) {
  Lattice canonical_lattice =
      xtal::canonical::equivalent(supercell.superlattice.superlattice(),
                                  supercell.prim->sym_info.point_group->element,
                                  supercell.superlattice.superlattice().tol());
  return std::make_shared<Supercell const>(
      supercell.prim, xtal::Superlattice(supercell.superlattice.prim_lattice(),
                                         canonical_lattice));
}

/// \brief Return SymOp that makes a supercell lattice canonical
///
/// The result, `op`, is the first in
/// `supercell.prim->sym_info.point_group->element` that satisfies:
///     canonical_supercell->superlattice.superlattice() ==
///         copy_apply(op, supercell.superlattice.superlattice())
SymOp to_canonical(Supercell const &supercell) {
  auto to_canonical_ix = xtal::canonical::operation_index(
      supercell.superlattice.superlattice(),
      supercell.prim->sym_info.point_group->element);
  return supercell.prim->sym_info.point_group->element[to_canonical_ix];
}

/// \brief Return op that makes a supercell lattice from the canonical
///     supercell lattice
///
/// The result, `op`, is the first in
/// `supercell.prim->sym_info.point_group->element` that satisfies:
///     supercell.superlattice.superlattice() ==
///         copy_apply(op, canonical_supercell->superlattice.superlattice())
SymOp from_canonical(Supercell const &supercell) {
  Lattice const &lhs = supercell.superlattice.superlattice();
  Lattice canonical =
      xtal::canonical::equivalent(supercell.superlattice.superlattice(),
                                  supercell.prim->sym_info.point_group->element,
                                  supercell.superlattice.superlattice().tol());
  for (auto const &op : supercell.prim->sym_info.point_group->element) {
    if (lhs == sym::copy_apply(op, canonical)) {
      return op;
    }
  }
  throw std::runtime_error(
      "Error in CASM::config::from_canonical(Supercell const &): Not found");
}

/// \brief Return the supercell with distinct symmetrically equivalent lattices
///
/// The results, `equiv`, are the distinct supercell with lattices generated by
/// `equiv = copy_apply(op, supercell.superlattice.lattice())` for `op` in
/// ``supercell.prim->sym_info.point_group->element`.
std::vector<std::shared_ptr<Supercell const>> make_equivalents(
    Supercell const &supercell) {
  std::set<Lattice> superlats;
  std::shared_ptr<Prim const> prim = supercell.prim;
  Lattice const &init_superlat = supercell.superlattice.superlattice();
  std::vector<SymOp> const &point_group = prim->sym_info.point_group->element;

  // function puts equivalent superlattice into canonical form
  auto representation_prepare = [&](Lattice const &superlat) {
    std::vector<Index> invariant_subgroup_indices =
        xtal::invariant_subgroup_indices(superlat, point_group);
    std::vector<SymOp> invariant_subgroup;
    for (Index i : invariant_subgroup_indices) {
      invariant_subgroup.push_back(point_group[i]);
    }
    return xtal::canonical::equivalent(superlat, invariant_subgroup);
  };

  // generate different orientations of equivalent superlattices
  for (auto const &op : point_group) {
    superlats.emplace(
        representation_prepare(sym::copy_apply(op, init_superlat)));
  }

  // make as shared Supercell
  std::vector<std::shared_ptr<Supercell const>> result;
  for (Lattice const &superlat : superlats) {
    result.push_back(std::make_shared<Supercell const>(prim, superlat));
  }

  return result;
}

/// \brief Return true if the operation does not mix given sites and other sites
bool site_indices_are_invariant(SupercellSymOp const &op,
                                std::set<Index> const &site_indices) {
  // Applying the operation indicated by `op` moves the value from
  // site index `op.permute_index(s)` to site index `s`, for each `s` in
  // the set. Therefore, if none of `op.permute_index(s)` are outside the
  // set `site_indices` the sites are invariant.

  return std::none_of(site_indices.begin(), site_indices.end(), [&](Index s) {
    return site_indices.count(op.permute_index(s)) == 0;
  });
}

}  // namespace config
}  // namespace CASM
