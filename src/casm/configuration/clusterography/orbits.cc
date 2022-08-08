#include "casm/configuration/clusterography/orbits.hh"

#include "casm/configuration/clusterography/ClusterInvariants.hh"
#include "casm/configuration/clusterography/ClusterSpecs.hh"
#include "casm/configuration/clusterography/IntegralCluster.hh"
#include "casm/configuration/clusterography/SubClusterCounter.hh"
#include "casm/configuration/group/Group.hh"
#include "casm/configuration/group/orbits.hh"
#include "casm/configuration/group/subgroups.hh"
#include "casm/configuration/sym_info/unitcellcoord_sym_info.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"
#include "casm/misc/algorithm.hh"

namespace CASM {
namespace clust {

/// \brief Copy cluster and apply symmetry operation transformation
///
/// \param op, Symmetry operation representation to be applied
/// \param clust, Cluster to transform
///
/// \return cluster, sorted and translated to the origin unit cell
///     after applying the symmetry operation transformation
IntegralCluster prim_periodic_integral_cluster_copy_apply(
    xtal::UnitCellCoordRep const &op, IntegralCluster clust) {
  if (!clust.size()) {
    return clust;
  }
  apply(op, clust);
  clust.sort();
  clust -= clust[0].unitcell();
  return clust;
}

/// \brief Find translation that leave cluster sites invariant after
///     transformation, up to a permutation
///
/// \param op, Symmetry operation representation to be applied
/// \param clust, Cluster to transform
///
/// \return translation, such that translation * op * clust is a
///     cluster with the same sites as the original clust, up to
///     a permutation
xtal::UnitCell prim_periodic_integral_cluster_frac_translation(
    xtal::UnitCellCoordRep const &op, IntegralCluster clust) {
  if (!clust.size()) {
    return xtal::UnitCell(0, 0, 0);
  }
  clust.sort();
  xtal::UnitCell pos_init = clust[0].unitcell();
  apply(op, clust);
  clust.sort();
  xtal::UnitCell pos_final = clust[0].unitcell();
  return pos_init - pos_final;
}

/// \brief Make an orbit of clusters, with periodic symmetry of a prim
///
/// \param orbit_element One cluster in the orbit
/// \param unitcellcoord_symgroup_rep Symmetry group representation (as
/// xtal::UnitCellCoordRep)
///
std::set<IntegralCluster> make_prim_periodic_orbit(
    IntegralCluster const &orbit_element,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep) {
  return group::make_orbit(orbit_element, unitcellcoord_symgroup_rep.begin(),
                           unitcellcoord_symgroup_rep.end(),
                           std::less<IntegralCluster>(),
                           prim_periodic_integral_cluster_copy_apply);
}

/// \brief Make groups that leave cluster orbit elements invariant
///
/// \param orbit A cluster orbit
/// \param factor_group The factor group used to generate the orbit.
/// \param lat_column_mat The 3x3 matrix whose columns are the lattice vectors.
/// \param unitcellcoord_symgroup_rep Symmetry group representation (as
///     xtal::UnitCellCoordRep) of the factor group.
///
/// \returns Cluster invariant groups, where cluster_groups[i] is
///     the SymGroup whose operations leave the sites of the i-th cluster in the
///     orbit invariant (up to a permutation).
std::vector<std::shared_ptr<SymGroup const>> make_cluster_groups(
    std::set<IntegralCluster> const &orbit,
    std::shared_ptr<SymGroup const> const &factor_group,
    Eigen::Matrix3d const &lat_column_mat,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep) {
  // The indices eq_map[i] are the indices of the group
  // elements transform the first element in the orbit into the
  // i-th element in the orbit.
  std::vector<std::vector<Index>> eq_map =
      group::make_equivalence_map(orbit, unitcellcoord_symgroup_rep.begin(),
                                  unitcellcoord_symgroup_rep.end(),
                                  prim_periodic_integral_cluster_copy_apply);

  // The indices subgroup_indices[i] are the indices of the group
  // elements which leave orbit element i invariant (up to a translation).
  std::vector<group::SubgroupIndices> subgroup_indices =
      group::make_invariant_subgroups(eq_map, *factor_group);

  // The group cluster_groups[i] contains the SymOp corresponding to
  // subgroup_indices[i] and including the translation which keeps
  // the i-th cluster invariant
  std::vector<std::shared_ptr<SymGroup const>> cluster_groups;
  auto orbit_it = orbit.begin();
  auto subgroup_indices_it = subgroup_indices.begin();
  auto subgroup_indices_end = subgroup_indices.end();

  // return xtal::SymOp, translation * factor_group->element[j], which leaves
  // *orbit_it invariant
  auto make_cluster_group_element = [&](Index j) {
    return xtal::SymOp(
               Eigen::Matrix3d::Identity(),
               lat_column_mat * prim_periodic_integral_cluster_frac_translation(
                                    unitcellcoord_symgroup_rep[j], *orbit_it)
                                    .cast<double>(),
               false) *
           factor_group->element[j];
  };

  while (subgroup_indices_it != subgroup_indices_end) {
    std::vector<xtal::SymOp> cluster_group_elements;
    for (Index j : *subgroup_indices_it) {
      cluster_group_elements.push_back(make_cluster_group_element(j));
    }
    cluster_groups.emplace_back(std::make_shared<SymGroup>(
        factor_group, cluster_group_elements, *subgroup_indices_it));
    ++subgroup_indices_it;
    ++orbit_it;
  }
  return cluster_groups;
}

/// \brief Make the group which leaves a cluster invariant
std::shared_ptr<SymGroup const> make_cluster_group(
    IntegralCluster cluster,
    std::shared_ptr<SymGroup const> const &factor_group,
    Eigen::Matrix3d const &lat_column_mat,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep) {
  if (!cluster.size()) {
    return factor_group;
  }

  cluster.sort();

  std::vector<xtal::SymOp> elements;
  std::set<Index> indices;
  for (Index i = 0; i < factor_group->element.size(); ++i) {
    IntegralCluster tclust = copy_apply(unitcellcoord_symgroup_rep[i], cluster);
    tclust.sort();

    xtal::UnitCell frac_trans = cluster[0].unitcell() - tclust[0].unitcell();
    tclust += frac_trans;

    if (tclust == cluster) {
      xtal::SymOp cart_trans(Eigen::Matrix3d::Identity(),
                             lat_column_mat * frac_trans.cast<double>(), false);
      elements.push_back(cart_trans * factor_group->element[i]);
      indices.insert(i);
    }
  }
  return std::make_shared<SymGroup>(factor_group, elements, indices);
}

/// \brief Make orbits of clusters, with periodic symmetry of a prim
///
/// \param prim The prim
/// \param unitcellcoord_symgroup_rep Symmetry representation for
///     transforming xtal::UnitCellCoord
/// \param site_filter Function that returns true if a xtal::Site
///     should be included in the generated clusters
/// \param max_length The value `max_length[branch]` is the
///     maximum site-to-site distance for clusters of size == branch.
///     The values for `branch==0` and `branch==1` are ignored. The
///     size of max_length sets the maximum number of sites
///     in a cluster.
/// \param custom_generators A vector of custom clusters to be
///     included regardless of site_filter and max_length. Includes
///     an option to specify that subclusters should also be included.
///
/// To generate `unitcellcoord_symgroup_rep`:
/// \code
/// // std::shared_ptr<SymGroup const> prim_factor_group;
/// std::vector<xtal::UnitCellCoordRep> unitcellcoord_symgroup_rep =
///     sym_info::make_unitcellcoord_symgroup_rep(
///         prim_factor_group->element, *prim);
/// \endcode
///
std::vector<std::set<IntegralCluster>> make_prim_periodic_orbits(
    std::shared_ptr<xtal::BasicStructure const> const &prim,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep,
    SiteFilterFunction site_filter, std::vector<double> const &max_length,
    std::vector<IntegralClusterOrbitGenerator> const &custom_generators) {
  // collect unique orbit elements, orbit branch by orbit branch
  typedef std::pair<ClusterInvariants, IntegralCluster> pair_type;
  CompareCluster_f compare_f(prim->lattice().tol());
  std::set<pair_type, CompareCluster_f> final(compare_f);
  std::set<pair_type, CompareCluster_f> prev_branch(compare_f);

  // include null cluster (it has been the convention in CASM)
  IntegralCluster null_cluster;
  final.emplace(ClusterInvariants(null_cluster, *prim), null_cluster);
  prev_branch.emplace(ClusterInvariants(null_cluster, *prim), null_cluster);

  // function to make a cluster canonical
  auto _make_canonical = [&](IntegralCluster const &cluster) {
    return group::make_canonical_element(
        cluster, unitcellcoord_symgroup_rep.begin(),
        unitcellcoord_symgroup_rep.end(), std::less<IntegralCluster>(),
        prim_periodic_integral_cluster_copy_apply);
  };

  for (int branch = 1; branch < max_length.size(); ++branch) {
    // generate candidate sites to be added to clusters of the previous branch
    std::vector<xtal::UnitCellCoord> candidate_sites;
    CandidateSitesFunction f;
    if (branch == 1) {
      f = origin_neighborhood();
    } else {
      f = max_length_neighborhood(max_length[branch]);
    }
    candidate_sites = f(*prim, site_filter);

    // a filter function selects which clusters are allowed
    ClusterFilterFunction cluster_filter;
    if (branch == 1) {
      cluster_filter = all_clusters_filter();
    } else {
      cluster_filter = max_length_cluster_filter(max_length[branch]);
    }

    // loop over clusters from the previous branch and add one site
    // keep the cluster if it passes the cluster filter and is unique
    std::set<pair_type, CompareCluster_f> curr_branch(compare_f);
    for (auto const &pair : prev_branch) {
      for (auto const &integral_site : candidate_sites) {
        IntegralCluster test_cluster = pair.second;
        if (CASM::contains(test_cluster.elements(), integral_site)) {
          continue;
        }
        test_cluster.elements().push_back(integral_site);
        ClusterInvariants invariants(test_cluster, *prim);
        if (!cluster_filter(invariants, test_cluster)) {
          continue;
        }
        test_cluster = _make_canonical(test_cluster);
        curr_branch.emplace(std::move(invariants), std::move(test_cluster));
      }
    }

    // save the previous branch
    final.insert(prev_branch.begin(), prev_branch.end());

    // the current branch becomes the previous branch
    prev_branch = std::move(curr_branch);
  }

  // save the last branch
  final.insert(prev_branch.begin(), prev_branch.end());

  // add custom generators -- filters do not apply
  for (auto const &custom_generator : custom_generators) {
    auto const &prototype = custom_generator.prototype;

    IntegralCluster test_cluster = _make_canonical(prototype);
    final.emplace(ClusterInvariants(test_cluster, *prim),
                  std::move(test_cluster));

    if (custom_generator.include_subclusters) {
      SubClusterCounter counter(prototype);
      Index i = 0;
      while (counter.valid()) {
        IntegralCluster test_cluster = _make_canonical(counter.value());
        final.emplace(ClusterInvariants(test_cluster, *prim),
                      std::move(test_cluster));
        counter.next();
      }
    }
  }

  // generate orbits from the unique clusters
  std::vector<std::set<IntegralCluster>> orbits;
  for (auto const &pair : final) {
    orbits.emplace_back(
        make_prim_periodic_orbit(pair.second, unitcellcoord_symgroup_rep));
  }

  return orbits;
}

// --- Local-cluster orbits ---

/// \brief Copy cluster and apply symmetry operation transformation
///
/// \param op, Symmetry operation representation to be applied
/// \param clust, Cluster to transform
///
/// \return cluster, sorted after applying the symmetry operation
///     transformation
IntegralCluster local_integral_cluster_copy_apply(
    xtal::UnitCellCoordRep const &op, IntegralCluster clust) {
  if (!clust.size()) {
    return clust;
  }
  apply(op, clust);
  clust.sort();
  return clust;
}

/// \brief Make an orbit of local clusters
///
/// \param orbit_element One cluster in the orbit
/// \param unitcellcoord_symgroup_rep Symmetry group representation (as
/// xtal::UnitCellCoordRep)
///
std::set<IntegralCluster> make_local_orbit(
    IntegralCluster const &orbit_element,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep) {
  return group::make_orbit(orbit_element, unitcellcoord_symgroup_rep.begin(),
                           unitcellcoord_symgroup_rep.end(),
                           std::less<IntegralCluster>(),
                           local_integral_cluster_copy_apply);
}

/// \brief Make groups that leave cluster orbit elements invariant
///
/// \param orbit A cluster orbit
/// \param head_group The phenomenal cluster group used to generate the orbit.
/// \param unitcellcoord_symgroup_rep Symmetry group representation (as
/// xtal::UnitCellCoordRep)
///
/// \returns Cluster invariant groups, where prim_periodic_cluster_groups[i], is
///     the SymGroup whose operations leave the sites of the i-th cluster in the
///     orbit invariant (up to a permutation).
std::vector<std::shared_ptr<SymGroup const>> make_local_cluster_groups(
    std::set<IntegralCluster> const &orbit,
    std::shared_ptr<SymGroup const> const &phenomenal_group,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep) {
  // The indices eq_map[i] are the indices of the group
  // elements transform the first element in the orbit into the
  // i-th element in the orbit.
  std::vector<std::vector<Index>> eq_map = group::make_equivalence_map(
      orbit, unitcellcoord_symgroup_rep.begin(),
      unitcellcoord_symgroup_rep.end(), local_integral_cluster_copy_apply);

  // The indices subgroup_indices[i] are the indices of the group
  // elements which leave orbit element i invariant.
  std::vector<group::SubgroupIndices> subgroup_indices =
      make_invariant_subgroups(eq_map, *phenomenal_group);

  // The group cluster_groups[i] contains the SymOp corresponding to
  // subgroup_indices[i].
  std::vector<std::shared_ptr<SymGroup const>> cluster_groups;
  auto orbit_it = orbit.begin();
  auto subgroup_indices_it = subgroup_indices.begin();
  auto subgroup_indices_end = subgroup_indices.end();
  while (subgroup_indices_it != subgroup_indices_end) {
    std::vector<xtal::SymOp> cluster_group_elements;
    for (Index j : *subgroup_indices_it) {
      cluster_group_elements.push_back(phenomenal_group->element[j]);
    }
    cluster_groups.emplace_back(std::make_shared<SymGroup>(
        phenomenal_group, cluster_group_elements, *subgroup_indices_it));
    ++subgroup_indices_it;
    ++orbit_it;
  }
  return cluster_groups;
}

/// \brief Make local-cluster orbits
///
/// \param prim The prim
/// \param unitcellcoord_symgroup_rep Symmetry representation for
///     transforming xtal::UnitCellCoord. This should agree with
///     phenomenal, being the cluster group or a subgroup
///     (currently no validation is performed).
/// \param site_filter Function that returns true if a xtal::Site
///     should be included in the generated clusters
/// \param max_length The value `max_length[branch]` is the
///     maximum site-to-site distance for clusters of size == branch.
///     The values for `branch==0` and `branch==1` are ignored. The
///     size of max_length sets the maximum number of sites
///     in a cluster.
/// \param custom_generators A vector of custom clusters to be
///     included regardless of site_filter and max_length. Includes
///     an option to specify that subclusters should also be included.
/// \param phenomenal The cluster around which local clusters are
///     generated.
/// \param cutoff_radius The value `cutoff_radius[branch]` is the
///     maximum phenomenal-site-to-cluster-site distance for clusters
///     of size == branch. The value for `branch==0` is ignored.
/// \param include_phenomenal_sites If true, include the phenomenal
///     cluster sites in the local clusters (default=false).
///
/// Often the easiest way to generate `unitcellcoord_symgroup_rep` consistent
/// with `phenomenal`, is to choose a phenomenal cluster from a cluster
/// orbit generated according to the periodic symmetry of the prim:
/// \code
/// // std::shared_ptr<xtal::BasicStructure const> prim;
/// // std::shared_ptr<SymGroup const> prim_factor_group;
/// // IntegralCluster phenomenal_prototype;
///
/// auto factor_group_unitcellcoord_symgroup_rep =
///     sym_info::make_unitcellcoord_symgroup_rep(
///         prim_factor_group->element, *prim);
/// auto prim_periodic_orbit =
///     make_prim_periodic_orbit(
///          phenomenal_prototype,
///          factor_group_unitcellcoordrep_symgroup_rep);
/// auto cluster_groups =
///     make_cluster_groups(
///         prim_periodic_orbit,
///         prim_factor_group,
///         prim->lattice().lat_column_mat(),
///         factor_group_unitcellcoord_symgroup_rep);
///
/// IntegralCluster phenomenal = *prim_periodic_orbit.begin();
/// auto unitcellcoord_symgroup_rep =
///     sym_info::make_unitcellcoord_symgroup_rep(
///         cluster_groups.begin()->element, *prim);
/// \endcode
///
/// To generate `unitcellcoord_symgroup_rep` for an arbitrary `phenomenal`
/// cluster:
/// \code
/// // std::shared_ptr<xtal::BasicStructure const> prim;
/// // std::shared_ptr<SymGroup const> prim_factor_group;
/// // IntegralCluster phenomenal;
/// auto factor_group_unitcellcoord_symgroup_rep =
///     sym_info::make_unitcellcoord_symgroup_rep(
///         prim_factor_group->element, *prim);
/// auto cluster_group = make_cluster_group(
///     phenomenal,
///     prim_factor_group,
///     prim->lattice().lat_column_mat(),
///     factor_group_unitcellcoord_symgroup_rep);
/// auto unitcellcoord_symgroup_rep =
///     sym_info::make_unitcellcoord_symgroup_rep(
///         cluster_group->element, *prim);
/// \endcode
///
std::vector<std::set<IntegralCluster>> make_local_orbits(
    std::shared_ptr<xtal::BasicStructure const> const &prim,
    std::vector<xtal::UnitCellCoordRep> const &unitcellcoord_symgroup_rep,
    SiteFilterFunction site_filter, std::vector<double> const &max_length,
    std::vector<IntegralClusterOrbitGenerator> const &custom_generators,
    IntegralCluster const &phenomenal, std::vector<double> const &cutoff_radius,
    bool include_phenomenal_sites) {
  // collect unique orbit elements, orbit branch by orbit branch
  typedef std::pair<ClusterInvariants, IntegralCluster> pair_type;
  CompareCluster_f compare_f(prim->lattice().tol());
  std::set<pair_type, CompareCluster_f> final(compare_f);
  std::set<pair_type, CompareCluster_f> prev_branch(compare_f);

  // include null cluster (it has been the convention in CASM)
  IntegralCluster null_cluster;
  final.emplace(ClusterInvariants(null_cluster, phenomenal, *prim),
                null_cluster);
  prev_branch.emplace(ClusterInvariants(null_cluster, phenomenal, *prim),
                      null_cluster);

  // function to make a cluster canonical
  auto _make_canonical = [&](IntegralCluster const &cluster) {
    return group::make_canonical_element(
        cluster, unitcellcoord_symgroup_rep.begin(),
        unitcellcoord_symgroup_rep.end(), std::less<IntegralCluster>(),
        local_integral_cluster_copy_apply);
  };

  for (int branch = 1; branch < max_length.size(); ++branch) {
    // generate candidate sites to be added to clusters of the previous branch
    CandidateSitesFunction f = cutoff_radius_neighborhood(
        phenomenal, cutoff_radius[branch], include_phenomenal_sites);
    std::vector<xtal::UnitCellCoord> candidate_sites = f(*prim, site_filter);

    // a filter function selects which clusters are allowed
    ClusterFilterFunction cluster_filter;
    if (branch == 1) {
      cluster_filter = all_clusters_filter();
    } else {
      cluster_filter = max_length_cluster_filter(max_length[branch]);
    }

    // loop over clusters from the previous branch and add one site
    // keep the cluster if it passes the cluster filter and is unique
    std::set<pair_type, CompareCluster_f> curr_branch(compare_f);
    for (auto const &pair : prev_branch) {
      for (auto const &integral_site : candidate_sites) {
        IntegralCluster test_cluster = pair.second;
        if (CASM::contains(test_cluster.elements(), integral_site)) {
          continue;
        }
        test_cluster.elements().push_back(integral_site);
        ClusterInvariants invariants(test_cluster, phenomenal, *prim);
        if (!cluster_filter(invariants, test_cluster)) {
          continue;
        }
        test_cluster = _make_canonical(test_cluster);
        curr_branch.emplace(std::move(invariants), std::move(test_cluster));
      }
    }

    // save the previous branch
    final.insert(prev_branch.begin(), prev_branch.end());

    // the current branch becomes the previous branch
    prev_branch = std::move(curr_branch);
  }

  // save the last branch
  final.insert(prev_branch.begin(), prev_branch.end());

  // add custom generators -- filters do not apply
  for (auto const &custom_generator : custom_generators) {
    auto const &prototype = custom_generator.prototype;

    IntegralCluster test_cluster = _make_canonical(prototype);
    final.emplace(ClusterInvariants(test_cluster, phenomenal, *prim),
                  std::move(test_cluster));

    if (custom_generator.include_subclusters) {
      SubClusterCounter counter(prototype);
      while (counter.valid()) {
        IntegralCluster test_cluster = _make_canonical(counter.value());
        final.emplace(ClusterInvariants(test_cluster, phenomenal, *prim),
                      std::move(test_cluster));
        counter.next();
      }
    }
  }

  // generate orbits from the unique clusters
  std::vector<std::set<IntegralCluster>> orbits;
  for (auto const &pair : final) {
    orbits.emplace_back(
        make_local_orbit(pair.second, unitcellcoord_symgroup_rep));
  }

  return orbits;
}

}  // namespace clust
}  // namespace CASM