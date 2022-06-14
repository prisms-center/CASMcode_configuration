#include "casm/configuration/SupercellSymInfo.hh"

#include "casm/configuration/Prim.hh"
#include "casm/crystallography/LinearIndexConverter.hh"
#include "casm/crystallography/Superlattice.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"

namespace CASM {
namespace config {

namespace SupercellSymInfo_impl {

/// \brief Construct supercell factor group
SymGroup make_factor_group(std::shared_ptr<Prim const> const &prim,
                           Superlattice const &superlattice) {
  std::vector<Index> invariant_subgroup_indices =
      xtal::invariant_subgroup_indices(superlattice.superlattice(),
                                       prim->sym_info.factor_group->element);
  std::set<Index> head_group_index(invariant_subgroup_indices.begin(),
                                   invariant_subgroup_indices.end());

  return SymGroup(prim->sym_info.factor_group, head_group_index);
}

/// \brief Construct supercell factor group permutations
///
/// These permutations describe how the translations within the supercell
/// permute sites in the supercell.
///
/// \brief ijk_index_converter UnitCell and linear unit cell index conversions
///     in this supercell. Generates translations within the supercell.
/// \brief bijk_index_converter UnitCellCoord and linear site index conversions
///     in this supercell.
std::vector<Permutation> make_translation_permutations(
    xtal::UnitCellIndexConverter const &ijk_index_converter,
    xtal::UnitCellCoordIndexConverter const &bijk_index_converter) {
  std::vector<Permutation> translation_permutations;

  // Loops over lattice points
  for (Index translation_ix = 0;
       translation_ix < ijk_index_converter.total_sites(); ++translation_ix) {
    std::vector<Index> single_translation_permutation(
        bijk_index_converter.total_sites(), -1);
    UnitCell translation_uc = ijk_index_converter(translation_ix);

    // Loops over all the sites
    for (Index old_site_ix = 0;
         old_site_ix < bijk_index_converter.total_sites(); ++old_site_ix) {
      UnitCellCoord old_site_ucc = bijk_index_converter(old_site_ix);
      Index new_site_ix = bijk_index_converter(old_site_ucc + translation_uc);

      single_translation_permutation[new_site_ix] = old_site_ix;
    }
    // You should have given a permutation value to every single site
    assert(std::find(single_translation_permutation.begin(),
                     single_translation_permutation.end(),
                     -1) == single_translation_permutation.end());
    translation_permutations.push_back(single_translation_permutation);
  }
  return translation_permutations;
}

/// \brief Construct supercell factor group permutations
///
/// These permutations describe how the prim factor group operations that are
/// consistent with this supercell permute sites in the supercell.
///
/// \brief head_group_index Indices in prim factor group of the supercell
///     factor group operations. Used as indices into
///     `basis_permutation_symgroup_rep`.
/// \brief basis_permutation_symgroup_rep Symmetry representation used to
///     transform integral site coordinates for this prim.
/// \brief bijk_index_converter UnitCellCoord and linear site index conversions
///     in this supercell
std::vector<Permutation> make_factor_group_permutations(
    std::vector<Index> const &head_group_index,
    BasisPermutationSymGroupRep const &basis_permutation_symgroup_rep,
    xtal::UnitCellCoordIndexConverter const &bijk_index_converter) {
  std::vector<Permutation> factor_group_permutations;
  long total_sites = bijk_index_converter.total_sites();

  for (Index operation_ix : head_group_index) {
    auto const &rep = basis_permutation_symgroup_rep[operation_ix];
    std::vector<Index> permutation(total_sites);
    for (Index old_l = 0; old_l < total_sites; ++old_l) {
      UnitCellCoord const &old_ucc = bijk_index_converter(old_l);
      UnitCellCoord new_ucc = copy_apply(rep, old_ucc);
      Index new_l = bijk_index_converter(new_ucc);
      permutation[new_l] = old_l;
    }
    factor_group_permutations.push_back(permutation);
  }
  return factor_group_permutations;
}

}  // namespace SupercellSymInfo_impl

/// \brief Constructor
///
/// \brief prim Prim associated with this supercell
/// \brief superlattice Superlattice for this supercell
/// \brief ijk_index_converter UnitCell and linear unit cell index conversions
///     in this supercell.
/// \brief bijk_index_converter UnitCellCoord and linear site index conversions
///     in this supercell.
SupercellSymInfo::SupercellSymInfo(
    std::shared_ptr<Prim const> const &prim, Superlattice const &superlattice,
    xtal::UnitCellIndexConverter const &unitcell_index_converter,
    xtal::UnitCellCoordIndexConverter const &unitcellcoord_index_converter)
    : factor_group(std::make_shared<SymGroup const>(
          SupercellSymInfo_impl::make_factor_group(prim, superlattice))),
      translation_permutations(
          SupercellSymInfo_impl::make_translation_permutations(
              unitcell_index_converter, unitcellcoord_index_converter)),
      factor_group_permutations(
          SupercellSymInfo_impl::make_factor_group_permutations(
              factor_group->head_group_index,
              prim->sym_info.basis_permutation_symgroup_rep,
              unitcellcoord_index_converter)) {}

}  // namespace config
}  // namespace CASM
