#ifndef CASM_occ_events_OccEventCounter
#define CASM_occ_events_OccEventCounter

#include <optional>
#include <set>
#include <vector>

#include "casm/configuration/clusterography/IntegralCluster.hh"
#include "casm/configuration/clusterography/occ_counter.hh"
#include "casm/configuration/occ_events/OccEvent.hh"
#include "casm/configuration/occ_events/OccSystem.hh"
#include "casm/configuration/occ_events/misc/MultiStepMethod.hh"
#include "casm/container/Counter.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace occ_events {

struct OccEventCounterData;
struct OccEventCounterStateInfo;

/// \brief OccEventCounter parameters
///
/// Notes:
/// - These parameters can be used to filter which events are
///   generated.
/// - The number of events generated can be reduced by setting
///   some of these parameters to restrict:
///   - cluster
///   - initial occupation
///   - final occupation
///   - atom trajectory
/// - A number of filters are already implemented, and custom
///   filters can be implemented using std::function.
///
/// Summary of defaults:
/// - Require number of atom conservation. (require_atom_conservation=true)
/// - Require type conservation.
/// (require_chemical_type_conserving_trajectories=true)
/// - Require xtal::Molecule that are indivisible do not breakup.
///   (do_not_allow_indivisible_molecule_breakup=true)
/// - Skip events in which the molecule or atom on one site does
///   nothing. (allow_subcluster_events=false)
/// - Skip events in which a vacancy moves to another vacancy.
///   (allow_subcluster_events=false)
/// - Skip events in which occ_final < occ_init, lexicographically.
///   This avoid duplicating the forward and reverse of many events.
///   In some cases, when the final and intial occupation indices
///   are the same, both the forward and reverse of an event will
///   still be generated. (allow_reverse_occ = false)
/// - Skip events in which two atoms directly exchange positions.
///   (skip_direct_exchange = true)
///
/// What the defaults allow generating:
/// - Atom-vacancy exchange
/// - Multi-site events
/// - Atoms to move through each other or very close to each other
/// - Divisible molecules to breakup (i.e. dumbbells)
/// - Molecules to re-orient (if no direct exchange)
/// - Events that are symmetrically equivalent
///
struct OccEventCounterParameters {
  // --- Cluster checks ---

  std::optional<int> min_cluster_size;
  std::optional<int> max_cluster_size;
  std::optional<int> required_cluster_size;
  std::optional<std::set<int>> excluded_sublattices;
  std::optional<std::set<int>> required_sublattices;
  std::optional<int> n_required_sublattices;

  /// \brief Optional customizeable filter to skip or allow events
  ///     based on the cluster. Return true to allow,
  ///     false to skip.
  std::function<bool(OccEventCounterData const &)> cluster_filter;

  // --- Initial occupation checks ---

  // Filter events by how many atom/molecule/orientations
  // exist before the event occurs. Note that the counts
  // apply to the event as generated, not the canonical
  // form.

  /// \brief Require initial occupation has particular
  ///     occupation indices. If the cluster size does not match,
  ///     or the indices are out-of-range for the cluster
  ///     sites, the occupation is not allowed.
  std::optional<Eigen::VectorXi> required_occ_init;

  // Order determined by OccSystem::atom_name_list
  std::optional<Eigen::VectorXi> min_init_atom_count;
  std::optional<Eigen::VectorXi> max_init_atom_count;
  std::optional<Eigen::VectorXi> required_init_atom_count;

  // Order determined by OccSystem::chemical_name_list
  std::optional<Eigen::VectorXi> min_init_molecule_count;
  std::optional<Eigen::VectorXi> max_init_molecule_count;
  std::optional<Eigen::VectorXi> required_init_molecule_count;

  // Order determined by OccSystem::orientation_name_list
  std::optional<Eigen::VectorXi> min_init_orientation_count;
  std::optional<Eigen::VectorXi> max_init_orientation_count;
  std::optional<Eigen::VectorXi> required_init_orientation_count;

  /// \brief Optional customizeable filter to skip or allow events
  ///     based on the cluster and occ_init. Return true to allow,
  ///     false to skip.
  std::function<bool(OccEventCounterData const &)> occ_init_filter;

  // --- Initial + Final occupation checks ---

  /// \brief Require final occupation has particular
  ///     occupation indices. If the cluster size does not
  ///     match, or the indices are out-of-range for the
  ///     cluster sites, the occupation is not allowed.
  std::optional<Eigen::VectorXi> required_occ_final;

  // Order determined by OccSystem::atom_name_list
  std::optional<Eigen::VectorXi> min_final_atom_count;
  std::optional<Eigen::VectorXi> max_final_atom_count;
  std::optional<Eigen::VectorXi> required_final_atom_count;

  // Order determined by OccSystem::molecule_name_list
  std::optional<Eigen::VectorXi> min_final_molecule_count;
  std::optional<Eigen::VectorXi> max_final_molecule_count;
  std::optional<Eigen::VectorXi> required_final_molecule_count;

  // Order determined by OccSystem::orientation_name_list
  std::optional<Eigen::VectorXi> min_final_orientation_count;
  std::optional<Eigen::VectorXi> max_final_orientation_count;
  std::optional<Eigen::VectorXi> required_final_orientation_count;

  /// \brief Optional customizeable filter to skip or allow events
  ///     based on the cluster, occ_init, and occ_final. Return true
  ///     to allow, false to skip.
  std::function<bool(OccEventCounterData const &)> occ_final_filter;

  // --- Trajectory checks ---

  /// \brief Skip or allow events which can be generated
  ///     using a subcluster because (i) a site is vacant
  ///     before and after the event or (ii) any molecule
  ///     does not change sites, break up, or re-orient.
  bool allow_subcluster_events = false;

  /// \brief Skip or allow events in which molecules break up
  ///     (i.e. do not allow dumbbell interstitials to separate)
  bool do_not_allow_breakup = false;

  /// \brief Skip events in which two atoms directly exchange
  ///     sites. Do not skip atom-vacancy exchange.
  bool skip_direct_exchange = true;

  /// \brief Optional customizeable filter to skip or allow events
  ///     based on the cluster, occ_init, occ_final, position_init,
  ///     and position_final. Return true to allow, false to skip.
  std::function<bool(OccEventCounterData const &)> trajectory_filter;

  // --- debugging ---

  /// \brief If true, print information about which states are
  ///     allowed / not allowed and why
  ///
  /// Typically set with:
  /// \code
  /// std::shared_ptr<OccSystem const> system = ...
  /// OccEventCounterParameters params;
  /// params.print_state_info = OccEventCounterStateInfoPrinter(*system);
  /// \endcode
  ///
  /// This is implemented as a function so it can be customized.
  std::function<void(OccEventCounterStateInfo const &info)> print_state_info;

  /// \brief If true, save information about which states are
  ///     allowed / not allowed and why.
  ///
  /// A OccEventCounterStateInfo instance is saved into
  /// OccEventCounterData::info for each step, recording the current
  /// value (if initialized) of the cluster, occ_init, occ_final,
  /// position_init, position_final, occ_event, and if the value is
  /// allowed or why it is not allowed.
  bool save_state_info = false;

  /// \brief If true, allow occ_final < occ_init, otherwise
  ///     this is not allowed to skip generating an event
  ///     and its reverse event
  bool allow_reverse_occ = false;

  /// \brief Current implementation requires this is true.
  ///     Future implementation will allow moves to/from reservoir
  ///     by setting this to false.
  bool require_atom_conservation = true;

  /// \brief If not allowing molecule breakup or moves to/from reservoir,
  ///     requiring molecule conservation can allow skipping some
  ///     unallowed final occupations.
  bool require_molecule_conservation = false;

  /// \brief Skip trajectories in which the atom/molecule type
  ///     changes (should always be true except for debugging
  ///     purposes)
  bool require_chemical_type_conserving_trajectories = true;

  /// \brief Skip or allow events in which indivisible molecules
  ///     break up (should always be true except for debugging
  ///     purposes)
  bool do_not_allow_indivisible_molecule_breakup = true;

  /// \brief Skip events in which a molecule in the reservoir
  ///     remains in the reservoir (for debugging purposes...
  ///     this situation should not occur)
  bool require_no_molecules_remain_in_reservoir = true;
};

/// \brief Saves OccEventCounter state info for output / debugging purposes
struct OccEventCounterStateInfo {
  clust::IntegralCluster cluster;
  std::vector<int> occ_init;
  std::vector<int> occ_final;
  std::vector<OccPosition> position_init;
  std::vector<OccPosition> position_final;
  std::optional<OccEvent> occ_event;
  std::string fails;
};

/// \brief Data structure used internally by OccEventCounter
struct OccEventCounterData {
  /// \brief Defines OccPosition indices, helps check OccEvents
  std::shared_ptr<OccSystem const> system;

  /// \brief All clusters on which OccEvents will be generated
  std::vector<clust::IntegralCluster> prototypes;

  /// \brief Parameters controlling the OccEventCounter
  OccEventCounterParameters params;

  /// \brief Counts over prototype vector. Incremented in
  ///     the outer-most step (step index 3).
  Index prototype_index;

  /// \brief Current cluster on which OccEvents are being generated,
  ///     a copy of prototype[prototype_index].
  clust::IntegralCluster cluster;

  /// \brief Counter generates initial occupation states on
  ///     the cluster. Incremented in second-outer-most step
  ///     (step index 2).
  Counter<std::vector<int>> occ_init_counter;

  /// \brief Counter generates final occupation states on
  ///     the cluster. Incremented in third-outer-most step
  ///     (step index 1).
  Counter<std::vector<int>> occ_final_counter;

  /// \brief Initial OccPosition, determined directly from occ_init_counter.
  ///
  /// Notes:
  /// - May be atom positions or molecule positions, depending on
  ///   `params.enumerate_individual_atom_trajectories`
  std::vector<OccPosition> position_init;

  /// \brief A permutation of `position_init`. The trajectories of
  ///     atoms or molecules are generated as position_init[i] ->
  ///     position_init[i]. Permuted in the inner-most step (step
  ///     index 0). Some permutations are allowable and some are
  ///     not because they result in atom/molecule type change,
  ///     which is checked during the method. A parameter
  ///     (`allow_subcluster_events`) determines whether permutations
  ///     which leave some sites unchanged (subcluster events) are
  ///     allowed.
  ///
  std::vector<OccPosition> position_final;

  /// \brief Set to true when position_final has iterated through
  ///     all permutations (and back to the initial position)
  bool trajectory_finished;

  /// \brief Current OccEvent
  OccEvent occ_event;

  /// \brief Track info about allowed / not allowed states
  std::vector<OccEventCounterStateInfo> info;
};

/// \brief Count over potential OccEvent
///
/// Notes:
/// - Given a vector of clust::IntegralCluster, this method iterates over
///   each cluster and generates all possible OccEvent according to
///   user-specified parameters.
/// - The method consists of 4 steps in a nested loop:
///   - iterate over cluster prototypes (outer-most step)
///   - iterate over initial cluster occupation
///   - iterate over final cluster occupation
///   - iterate over initial->final trajectory permutations (inner-most step)
/// - At each step, a number of criteria, specified by
/// OccEventCounterParameters,
///   are checked to skip invalid or undesired OccEvent.
class OccEventCounter {
 public:
  /// \brief Constructor
  ///
  /// \param system, OccSystem used to define and check OccEvent
  /// \param clusters, Vector of underlying cluster orbit prototypes
  ///     on which OccEvent should be generated.
  /// \param params, Options controlling the events generated.
  ///
  OccEventCounter(std::shared_ptr<OccSystem const> const &system,
                  std::vector<clust::IntegralCluster> const &prototypes,
                  OccEventCounterParameters const &params);

  std::shared_ptr<OccEventCounterData> const &data() const;

  /// \brief Advance to the next allowed OccEvent
  bool advance();

  /// \brief Current OccEvent
  OccEvent const &value() const;

  /// \brief True if counting is finished
  bool is_finished() const;

 private:
  /// This holds current method state and parameters
  std::shared_ptr<OccEventCounterData> m_data;

  /// This implements the nested loop method:
  ///   - iterate over cluster prototypes (outer-most step)
  ///   - iterate over initial cluster occupation
  ///   - iterate over final cluster occupation
  ///   - iterate over initial->final trajectory permutations (inner-most step)
  /// - At each step, a number of criteria, specified by
  /// OccEventCounterParameters,
  ///   are checked to skip invalid or undesired OccEvent.
  std::unique_ptr<MultiStepMethod<OccEventCounterData>> m_stepper;
};

}  // namespace occ_events
}  // namespace CASM

#endif
