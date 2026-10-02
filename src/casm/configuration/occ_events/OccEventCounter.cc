#include "casm/configuration/occ_events/OccEventCounter.hh"

#include "casm/crystallography/BasicStructure.hh"

namespace CASM {
namespace occ_events {

namespace {

/// \brief Outer-most step: iterate over cluster prototypes
class PrototypeClusterCounter : public SingleStepBase<OccEventCounterData> {
 public:
  PrototypeClusterCounter(std::shared_ptr<OccEventCounterData> _data)
      : SingleStepBase<OccEventCounterData>(_data) {}

  /// \brief Advance state, return true if post-state is valid / not finished
  bool advance() override {
    if (is_finished()) {
      return false;
    }
    ++data()->prototype_index;
    bool result = !is_finished();
    if (result) {
      data()->cluster = data()->prototypes[data()->prototype_index];
    }
    return result;
  }

  /// \brief Return true if in invalid / finished state
  bool is_finished() const override {
    return data()->prototype_index >= data()->prototypes.size();
  }

  /// \brief Return true if in a not-finished && allowed state
  bool is_allowed() const override {
    if (this->fails_min_cluster_size()) {
      _fails("cluster:min_cluster_size");
      return false;
    }
    if (this->fails_max_cluster_size()) {
      _fails("cluster:max_cluster_size");
      return false;
    }
    if (this->fails_required_cluster_size()) {
      _fails("cluster:required_cluster_size");
      return false;
    }
    if (this->fails_excluded_sublattices()) {
      _fails("cluster:excluded_sublattices");
      return false;
    }
    if (this->fails_required_sublattices()) {
      _fails("cluster:required_sublattices");
      return false;
    }

    // customizeable filter
    if (data()->params.cluster_filter &&
        !data()->params.cluster_filter(*data())) {
      _fails("cluster:filter");
      return false;
    }

    _passes();
    return true;
  }

  void _fails(std::string what) const {
    if (!data()->params.save_state_info) {
      return;
    }
    OccEventCounterStateInfo state_info;
    state_info.cluster = data()->cluster;
    state_info.fails = what;
    data()->info.push_back(state_info);
  }

  void _passes() const {
    // continue;
  }

  /// \brief Check if cluster size is at minimum a
  ///     specified value (min_cluster_size)
  bool fails_min_cluster_size() const {
    if (!data()->params.min_cluster_size.has_value()) {
      return false;
    }
    return data()->cluster.size() < *data()->params.min_cluster_size;
  }

  /// \brief Check if cluster size is at maximum a
  ///     specified value (min_cluster_size)
  bool fails_max_cluster_size() const {
    if (!data()->params.max_cluster_size.has_value()) {
      return false;
    }
    return data()->cluster.size() > *data()->params.max_cluster_size;
  }

  /// \brief Check if cluster size is a
  ///     specified value (required_cluster_size)
  bool fails_required_cluster_size() const {
    if (!data()->params.required_cluster_size.has_value()) {
      return false;
    }
    return data()->cluster.size() != *data()->params.required_cluster_size;
  }

  /// \brief Check if cluster includes any
  ///     excluded sublattice (excluded_sublattices)
  bool fails_excluded_sublattices() const {
    if (!data()->params.excluded_sublattices.has_value()) {
      return false;
    }
    auto const &excluded_sublattices = *data()->params.excluded_sublattices;
    for (auto const &site : data()->cluster) {
      if (excluded_sublattices.count(site.sublattice())) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if cluster includes sufficient number of
  ///     required sublattice (required_sublattices)
  bool fails_required_sublattices() const {
    if (!data()->params.required_sublattices.has_value()) {
      return false;
    }
    auto const &required_sublattices = *data()->params.required_sublattices;
    std::map<int, bool> has_sublattice;
    for (auto const &b : required_sublattices) {
      has_sublattice[b] = false;
    }
    for (auto const &site : data()->cluster) {
      auto it = has_sublattice.find(site.sublattice());
      if (it != has_sublattice.end()) {
        it->second = true;
      }
    } 
    // require at least n_required_sublattices matches
    // of sublattices in required_sublattices
    int misses_remaining = 0; // default must match all (no misses)
    if (data()->params.n_required_sublattices.has_value()) {
      int n_required_sublattices = *data()->params.n_required_sublattices;
      misses_remaining =
          static_cast<int>(has_sublattice.size()) - n_required_sublattices;
    }
    for (auto const &pair : has_sublattice) {
      if (pair.second == false) {
        if (--misses_remaining < 0) {
          return true;
        }
      }
    }
    return false;
  }

  /// \brief Initialize `prototype_index=0` and set `cluster`
  void initialize() const override {
    data()->prototype_index = 0;
    if (!data()->prototypes.size()) {
      return;
    }
    data()->cluster = data()->prototypes[data()->prototype_index];
  }
};

/// \brief Iterate over initial cluster occupation
class OccInitCounter : public SingleStepBase<OccEventCounterData> {
 public:
  OccInitCounter(std::shared_ptr<OccEventCounterData> _data)
      : SingleStepBase<OccEventCounterData>(_data) {}

  /// \brief Advance state, return true if post-state is valid / not finished
  bool advance() override { return ++data()->occ_init_counter; }

  /// \brief Return true if in invalid / finished state
  bool is_finished() const override {
    return !data()->occ_init_counter.valid();
  }

  /// \brief Return true if in a not-finished && allowed state
  bool is_allowed() const override {
    // occ init
    if (this->fails_required_occ_init()) {
      _fails("occ_init:required_occ_init");
      return false;
    }

    // atom count
    if (this->fails_required_init_atom_count()) {
      _fails("occ_init:required_init_atom_count");
      return false;
    }
    if (this->fails_min_init_atom_count()) {
      _fails("occ_init:min_init_atom_count");
      return false;
    }
    if (this->fails_max_init_atom_count()) {
      _fails("occ_init:max_init_atom_count");
      return false;
    }

    // molecule count
    if (this->fails_required_init_molecule_count()) {
      _fails("occ_init:required_init_molecule_count");
      return false;
    }
    if (this->fails_min_init_molecule_count()) {
      _fails("occ_init:min_init_molecule_count");
      return false;
    }
    if (this->fails_max_init_molecule_count()) {
      _fails("occ_init:max_init_molecule_count");
      return false;
    }

    // orientation count
    if (this->fails_required_init_orientation_count()) {
      _fails("occ_init:required_init_orientation_count");
      return false;
    }
    if (this->fails_min_init_orientation_count()) {
      _fails("occ_init:min_init_orientation_count");
      return false;
    }
    if (this->fails_max_init_orientation_count()) {
      _fails("occ_init:max_init_orientation_count");
      return false;
    }

    // customizeable filter
    if (data()->params.occ_init_filter &&
        !data()->params.occ_init_filter(*data())) {
      _fails("occ_init:filter");
      return false;
    }

    _passes();
    return true;
  }

  void _fails(std::string what) const {
    if (!data()->params.print_state_info && !data()->params.save_state_info) {
      return;
    }
    OccEventCounterStateInfo state_info;
    state_info.cluster = data()->cluster;
    state_info.occ_init = data()->occ_init_counter();
    state_info.fails = what;
    if (data()->params.print_state_info) {
      data()->params.print_state_info(state_info);
    }
    if (data()->params.save_state_info) {
      data()->info.push_back(state_info);
    }
  }

  void _passes() const {
    // continue
  }

  /// \brief Check if initial occupation satisifies
  ///     required occupation (required_occ_init)
  bool fails_required_occ_init() const {
    if (!data()->params.required_occ_init.has_value()) {
      return false;
    }
    std::vector<int> const &occ_init = data()->occ_init_counter();
    Eigen::VectorXi const &required_occ_init =
        *data()->params.required_occ_init;
    if (occ_init.size() != required_occ_init.size()) {
      return true;
    }
    for (Index i = 0; i < data()->cluster.size(); ++i) {
      if (occ_init[i] != required_occ_init[i]) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     required (exact) atom count criteria (required_init_atom_count)
  bool fails_required_init_atom_count() const {
    if (!data()->params.required_init_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_init_counter());
    return m_count != *data()->params.required_init_atom_count;
  }

  /// \brief Check if initial occupation satisifies
  ///     minimum atom count criteria (min_init_atom_count)
  bool fails_min_init_atom_count() const {
    if (!data()->params.min_init_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_init_counter());
    Eigen::VectorXi min_count = *data()->params.min_init_atom_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error("Error: min_init_atom_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     maximum atom count criteria (max_init_atom_count)
  bool fails_max_init_atom_count() const {
    if (!data()->params.max_init_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_init_counter());
    Eigen::VectorXi max_count = *data()->params.max_init_atom_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error("Error: max_init_atom_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     required (exact) molecule count criteria
  ///     (required_init_molecule_count)
  bool fails_required_init_molecule_count() const {
    if (!data()->params.required_init_molecule_count.has_value()) {
      return false;
    }
    // data()->params.required_init_molecule_count->transpose() << std::endl;
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_init_counter());
    return m_count != *data()->params.required_init_molecule_count;
  }

  /// \brief Check if initial occupation satisifies
  ///     minimum molecule count criteria (min_init_molecule_count)
  bool fails_min_init_molecule_count() const {
    if (!data()->params.min_init_molecule_count.has_value()) {
      return false;
    }
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_init_counter());
    Eigen::VectorXi min_count = *data()->params.min_init_molecule_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error("Error: min_init_molecule_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     maximum molecule count criteria (max_init_molecule_count)
  bool fails_max_init_molecule_count() const {
    if (!data()->params.max_init_molecule_count.has_value()) {
      return false;
    }
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_init_counter());
    Eigen::VectorXi max_count = *data()->params.max_init_molecule_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error("Error: max_init_molecule_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     required (exact) orientation count criteria
  ///     (required_init_orientation_count)
  bool fails_required_init_orientation_count() const {
    if (!data()->params.required_init_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_init_counter());
    return m_count != *data()->params.required_init_orientation_count;
  }

  /// \brief Check if initial occupation satisifies
  ///     minimum orientation count criteria (min_init_orientation_count)
  bool fails_min_init_orientation_count() const {
    if (!data()->params.min_init_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_init_counter());
    Eigen::VectorXi min_count = *data()->params.min_init_orientation_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error(
          "Error: min_init_orientation_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if initial occupation satisifies
  ///     maximum orientation count criteria (max_init_orientation_count)
  bool fails_max_init_orientation_count() const {
    if (!data()->params.max_init_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_init_counter());
    Eigen::VectorXi max_count = *data()->params.max_init_orientation_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error(
          "Error: max_init_orientation_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Initialize `occ_init_counter` for current cluster
  void initialize() const override {
    data()->occ_init_counter =
        make_occ_counter(data()->cluster, *data()->system->prim);
  }

 private:
  /// \brief Temporary variable used for checking atom/molecule/orientation
  /// counts
  mutable Eigen::VectorXi m_count;
};

/// \brief Iterate over final cluster occupation
class OccFinalCounter : public SingleStepBase<OccEventCounterData> {
 public:
  OccFinalCounter(std::shared_ptr<OccEventCounterData> _data)
      : SingleStepBase<OccEventCounterData>(_data) {}

  /// \brief Advance state, return true if post-state is not finished
  bool advance() override { return ++data()->occ_final_counter; }

  /// \brief Return true if in finished state
  bool is_finished() const override {
    return !data()->occ_final_counter.valid();
  }

  /// \brief Return true if in a not-finished && allowed state
  bool is_allowed() const override {
    // occ final
    if (this->fails_required_occ_final()) {
      _fails("occ_init:required_occ_final");
      return false;
    }

    if (this->fails_allow_reverse_occ()) {
      _fails("occ_final:allow_reverse_occ");
      return false;
    }
    // std::cout << "OccFinalCounter is_allowed 1" << std::endl;
    if (this->fails_require_atom_conservation()) {
      _fails("occ_final:require_atom_conservation");
      return false;
    }
    // std::cout << "OccFinalCounter is_allowed 2" << std::endl;
    if (this->fails_require_molecule_conservation()) {
      _fails("occ_final:require_molecule_conservation");
      return false;
    }
    // std::cout << "OccFinalCounter is_allowed 3" << std::endl;
    if (this->fails_allow_subcluster_events()) {
      _fails("occ_final:allow_subcluster_events:occ_final");
      return false;
    }

    // atom count
    if (this->fails_required_final_atom_count()) {
      _fails("occ_final:required_final_atom_count");
      return false;
    }
    if (this->fails_min_final_atom_count()) {
      _fails("occ_final:min_final_atom_count");
      return false;
    }
    if (this->fails_max_final_atom_count()) {
      _fails("occ_final:max_final_atom_count");
      return false;
    }

    // molecule count
    if (this->fails_required_final_molecule_count()) {
      _fails("occ_final:required_final_molecule_count");
      return false;
    }
    if (this->fails_min_final_molecule_count()) {
      _fails("occ_final:min_final_molecule_count");
      return false;
    }
    if (this->fails_max_final_molecule_count()) {
      _fails("occ_final:max_final_molecule_count");
      return false;
    }

    // orientation count
    if (this->fails_required_final_orientation_count()) {
      _fails("occ_final:required_final_orientation_count");
      return false;
    }
    if (this->fails_min_final_orientation_count()) {
      _fails("occ_final:min_final_orientation_count");
      return false;
    }
    if (this->fails_max_final_orientation_count()) {
      _fails("occ_final:max_final_orientation_count");
      return false;
    }

    // customizeable filter
    if (data()->params.occ_final_filter &&
        !data()->params.occ_final_filter(*data())) {
      _fails("occ_final:filter");
      return false;
    }

    _passes();
    return true;
  }

  void _fails(std::string what) const {
    if (!data()->params.print_state_info && !data()->params.save_state_info) {
      return;
    }
    OccEventCounterStateInfo state_info;
    state_info.cluster = data()->cluster;
    state_info.occ_init = data()->occ_init_counter();
    state_info.occ_final = data()->occ_final_counter();
    state_info.fails = what;
    if (data()->params.print_state_info) {
      data()->params.print_state_info(state_info);
    }
    if (data()->params.save_state_info) {
      data()->info.push_back(state_info);
    }
  }

  void _passes() const {
    // continue
  }

  /// \brief Check if final occupation satisifies
  ///     required occupation (required_occ_final)
  bool fails_required_occ_final() const {
    if (!data()->params.required_occ_final.has_value()) {
      return false;
    }
    std::vector<int> const &occ_final = data()->occ_final_counter();
    Eigen::VectorXi const &required_occ_final =
        *data()->params.required_occ_final;
    if (occ_final.size() != required_occ_final.size()) {
      return true;
    }
    for (Index i = 0; i < data()->cluster.size(); ++i) {
      if (occ_final[i] != required_occ_final[i]) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if occ_final < occ_init to skip generating an event
  ///     and its reverse event (allow_reverse_occ)
  bool fails_allow_reverse_occ() const {
    if (data()->params.allow_reverse_occ == true) {
      return false;
    }
    return data()->occ_final_counter() < data()->occ_init_counter();
  }

  /// \brief Check if initial and final cluster occupation
  ///     have the same number of each type of atom (require_atom_conservation)
  bool fails_require_atom_conservation() const {
    if (!data()->params.require_atom_conservation) {
      return false;
    }
    return !data()->system->is_atom_conserving(m_count, data()->cluster,
                                               data()->occ_init_counter(),
                                               data()->occ_final_counter());
  }

  /// \brief Check if initial and final cluster occupation
  ///     have the same number of each type of molecule (may
  ///     have different orientations) (require_molecule_conservation)
  bool fails_require_molecule_conservation() const {
    if (!data()->params.require_molecule_conservation) {
      return false;
    }
    return !data()->system->is_molecule_conserving(m_count, data()->cluster,
                                                   data()->occ_init_counter(),
                                                   data()->occ_final_counter());
  }

  /// \brief Check if event can be described using a subcluster,
  ///     because a site is vacant before and after the event
  ///     (allow_subcluster_events)
  bool fails_allow_subcluster_events() const {
    return data()->params.allow_subcluster_events == false &&
           data()->system->is_any_unchanging_vacant_site(
               data()->cluster, data()->occ_init_counter(),
               data()->occ_final_counter());
  }

  /// \brief Check if final occupation satisifies
  ///     required (exact) atom count criteria (required_final_atom_count)
  bool fails_required_final_atom_count() const {
    if (!data()->params.required_final_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_final_counter());
    return m_count != *data()->params.required_final_atom_count;
  }

  /// \brief Check if final occupation satisifies
  ///     minimum atom count criteria (min_final_atom_count)
  bool fails_min_final_atom_count() const {
    if (!data()->params.min_final_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_final_counter());
    Eigen::VectorXi min_count = *data()->params.min_final_atom_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error("Error: min_final_atom_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if final occupation satisifies
  ///     maximum atom count criteria (max_final_atom_count)
  bool fails_max_final_atom_count() const {
    if (!data()->params.max_final_atom_count.has_value()) {
      return false;
    }
    data()->system->atom_count(m_count, data()->cluster,
                               data()->occ_final_counter());
    Eigen::VectorXi max_count = *data()->params.max_final_atom_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error("Error: max_final_atom_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if final occupation satisifies
  ///     required (exact) molecule count criteria
  ///     (required_final_molecule_count)
  bool fails_required_final_molecule_count() const {
    if (!data()->params.required_final_molecule_count.has_value()) {
      return false;
    }
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_final_counter());
    return m_count != *data()->params.required_final_molecule_count;
  }

  /// \brief Check if final occupation satisifies
  ///     minimum molecule count criteria (min_final_molecule_count)
  bool fails_min_final_molecule_count() const {
    if (!data()->params.min_final_molecule_count.has_value()) {
      return false;
    }
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_final_counter());
    Eigen::VectorXi min_count = *data()->params.min_final_molecule_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error("Error: min_final_molecule_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if final occupation satisifies
  ///     maximum molecule count criteria (max_final_molecule_count)
  bool fails_max_final_molecule_count() const {
    if (!data()->params.max_final_molecule_count.has_value()) {
      return false;
    }
    data()->system->molecule_count(m_count, data()->cluster,
                                   data()->occ_final_counter());
    Eigen::VectorXi max_count = *data()->params.max_final_molecule_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error("Error: max_final_molecule_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if final occupation satisifies
  ///     required (exact) orientation count criteria
  ///     (required_final_orientation_count)
  bool fails_required_final_orientation_count() const {
    if (!data()->params.required_final_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_final_counter());
    return m_count != *data()->params.required_final_orientation_count;
  }

  /// \brief Check if final occupation satisifies
  ///     minimum orientation count criteria (min_final_orientation_count)
  bool fails_min_final_orientation_count() const {
    if (!data()->params.min_final_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_final_counter());
    Eigen::VectorXi min_count = *data()->params.min_final_orientation_count;
    if (min_count.size() != m_count.size()) {
      throw std::runtime_error(
          "Error: min_final_orientation_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) < min_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Check if final occupation satisifies
  ///     maximum orientation count criteria (max_final_orientation_count)
  bool fails_max_final_orientation_count() const {
    if (!data()->params.max_final_orientation_count.has_value()) {
      return false;
    }
    data()->system->orientation_count(m_count, data()->cluster,
                                      data()->occ_final_counter());
    Eigen::VectorXi max_count = *data()->params.max_final_orientation_count;
    if (max_count.size() != m_count.size()) {
      throw std::runtime_error(
          "Error: max_final_orientation_count size mismatch");
    }
    for (Index i = 0; i < m_count.size(); ++i) {
      if (m_count(i) > max_count(i)) {
        return true;
      }
    }
    return false;
  }

  /// \brief Initialize `occ_final_counter` for current cluster
  void initialize() const override {
    data()->occ_final_counter =
        make_occ_counter(data()->cluster, *data()->system->prim);
  }

 private:
  /// \brief Temporary variable used for checking atom/molecule conservation
  mutable Eigen::VectorXi m_count;
};

/// \brief Inner-most step: iterate over OccPosition permutations
class TrajectoryCounter : public SingleStepBase<OccEventCounterData> {
 public:
  TrajectoryCounter(std::shared_ptr<OccEventCounterData> _data)
      : SingleStepBase<OccEventCounterData>(_data) {
    data()->trajectory_finished = true;
  }

  /// \brief Advance state, return true if post-state is not finished
  ///
  /// Notes:
  /// - Permutes `position_final` until no more permutations allowed
  bool advance() override {
    bool valid = std::next_permutation(data()->position_final.begin(),
                                       data()->position_final.end());
    if (valid) {
      data()->occ_event =
          make_occevent(data()->position_init, data()->position_final);
    } else {
      data()->position_init.clear();
      data()->position_final.clear();
      data()->trajectory_finished = true;
    }
    return !is_finished();
  }

  /// \brief Return true if in finished state
  bool is_finished() const override { return data()->trajectory_finished; }

  /// \brief Return true if in a not-finished && allowed state
  bool is_allowed() const override {
    if (this->fails_require_chemical_type_conserving_trajectories()) {
      _fails("trajectory:require_chemical_type_conserving_trajectories");
      return false;
    }
    if (this->fails_skip_direct_exchange()) {
      _fails("trajectory:skip_direct_exchange");
      return false;
    }
    if (this->fails_allow_subcluster_events()) {
      _fails("trajectory:allow_subcluster_events");
      return false;
    }
    if (this->fails_do_not_allow_breakup()) {
      _fails("trajectory:do_not_allow_breakup");
      return false;
    }
    if (this->fails_do_not_allow_indivisible_molecule_breakup()) {
      _fails("trajectory:do_not_allow_indivisible_molecule_breakup");
      return false;
    }
    if (this->fails_require_no_molecules_remain_in_reservoir()) {
      _fails("trajectory:require_no_molecules_remain_in_reservoir");
      return false;
    }

    // customizeable filter
    if (data()->params.trajectory_filter &&
        !data()->params.trajectory_filter(*data())) {
      _fails("trajectory:filter");
      return false;
    }

    _passes();
    return true;
  }

  void _fails(std::string what) const {
    if (!data()->params.print_state_info && !data()->params.save_state_info) {
      return;
    }
    OccEventCounterStateInfo state_info;
    state_info.cluster = data()->cluster;
    state_info.occ_init = data()->occ_init_counter();
    state_info.occ_final = data()->occ_final_counter();
    state_info.position_init = data()->position_init;
    state_info.position_final = data()->position_final;
    state_info.occ_event = data()->occ_event;
    state_info.fails = what;
    if (data()->params.print_state_info) {
      data()->params.print_state_info(state_info);
    }
    if (data()->params.save_state_info) {
      data()->info.push_back(state_info);
    }
  }

  void _passes() const {
    if (!data()->params.print_state_info && !data()->params.save_state_info) {
      return;
    }
    OccEventCounterStateInfo state_info;
    state_info.cluster = data()->cluster;
    state_info.occ_init = data()->occ_init_counter();
    state_info.occ_final = data()->occ_final_counter();
    state_info.position_init = data()->position_init;
    state_info.position_final = data()->position_final;
    state_info.occ_event = data()->occ_event;
    if (data()->params.print_state_info) {
      data()->params.print_state_info(state_info);
    }
    if (data()->params.save_state_info) {
      data()->info.push_back(state_info);
    }
  }

  /// \brief Check for trajectories in which the atom/molecule type
  ///     changes (should always be required except for debugging
  ///     purposes) (require_chemical_type_conserving_trajectories)
  bool fails_require_chemical_type_conserving_trajectories() const {
    return data()->params.require_chemical_type_conserving_trajectories ==
               true &&
           !data()->system->is_chemical_type_conserving(data()->position_init,
                                                        data()->position_final);
  }

  /// \brief Check for trajectories in which two atoms
  ///     directly exchange sites. Do not skip atom-vacancy exchange.
  ///     (skip_direct_exchange)
  bool fails_skip_direct_exchange() const {
    return data()->params.skip_direct_exchange &&
           data()->system->is_direct_exchange(data()->position_init,
                                              data()->position_final);
  }

  /// \brief Check if event can be described using a subcluster, because
  ///     any molecule does not change sites, break up, or re-orient
  ///     (allow_subcluster_events)
  bool fails_allow_subcluster_events() const {
    return data()->params.allow_subcluster_events == false &&
           data()->system->is_any_unchanging_molecule(data()->position_init,
                                                      data()->position_final);
  }

  /// \brief Check if event involves breaking up a molecule
  ///     (do_not_allow_breakup)
  bool fails_do_not_allow_breakup() const {
    return data()->params.do_not_allow_breakup == true &&
           data()->system->is_molecule_breakup(data()->position_init,
                                               data()->position_final);
  }

  /// \brief Check if event involves breaking up an indivisible molecule
  ///     (do_not_allow_indivisible_molecule_breakup)
  bool fails_do_not_allow_indivisible_molecule_breakup() const {
    return data()->params.do_not_allow_indivisible_molecule_breakup == true &&
           data()->system->is_indivisible_molecule_breakup(
               data()->position_init, data()->position_final);
  }

  /// \brief Check for events in which a molecule in the reservoir
  ///     remains in the reservoir (for debugging purposes...
  ///     this situation should not occur)
  ///     (require_no_molecules_remain_in_reservoir)
  bool fails_require_no_molecules_remain_in_reservoir() const {
    return data()->params.require_no_molecules_remain_in_reservoir == true &&
           data()->system->is_any_unchanging_reservoir_type(
               data()->position_init, data()->position_final);
  }

  /// \brief Initialize `position_init` and `position_final` based on
  ///     `occ_init` and `occ_final`, respectively. `position_final` is
  ///     sorted so that all permutations of `position_final` can be
  ///     generated.
  void initialize() const override {
    data()->trajectory_finished = false;

    data()->system->make_occ_positions(
        data()->position_init, m_count, data()->cluster,
        data()->occ_init_counter(), data()->occ_final_counter(),
        data()->params.require_atom_conservation);

    data()->system->make_occ_positions(
        data()->position_final, m_count, data()->cluster,
        data()->occ_final_counter(), data()->occ_init_counter(),
        data()->params.require_atom_conservation);

    std::sort(data()->position_final.begin(), data()->position_final.end());

    data()->occ_event =
        make_occevent(data()->position_init, data()->position_final);
  }

 private:
  /// \brief Temporary variable used for checking atom/molecule conservation
  mutable Eigen::VectorXi m_count;
};

}  // namespace

/// \brief Constructor
///
/// \param system, OccSystem used to define and check OccEvent
/// \param clusters, Vector of underlying cluster orbit prototypes
///     on which OccEvent should be generated.
/// \param params, Options controlling the events generated.
///
OccEventCounter::OccEventCounter(
    std::shared_ptr<OccSystem const> const &system,
    std::vector<clust::IntegralCluster> const &prototypes,
    OccEventCounterParameters const &params) {
  // make shared data structure
  m_data = std::make_shared<OccEventCounterData>();
  m_data->system = system;
  m_data->prototypes = prototypes;
  m_data->params = params;

  // make individual method steps:
  typedef MultiStepMethod<OccEventCounterData>::StepVector StepVector;
  StepVector steps;

  // inner-most step: iterate over OccPosition permutations
  steps.emplace_back(std::make_unique<TrajectoryCounter>(m_data));

  // iterate over final cluster occupation
  steps.emplace_back(std::make_unique<OccFinalCounter>(m_data));

  // iterate over initial cluster occupation
  steps.emplace_back(std::make_unique<OccInitCounter>(m_data));

  // outer-most step: iterate over cluster prototypes
  steps.emplace_back(std::make_unique<PrototypeClusterCounter>(m_data));

  // make MultiStepMethod, which will iterate through each individual step
  // in a Counter-like fashion, skipping invalid or unallowed events
  m_stepper = std::make_unique<MultiStepMethod<OccEventCounterData>>(
      m_data, std::move(steps));
}

std::shared_ptr<OccEventCounterData> const &OccEventCounter::data() const {
  return m_data;
}

/// \brief Advance to the next allowed OccEvent
bool OccEventCounter::advance() {
  m_stepper->advance();
  return !is_finished();
}

OccEvent const &OccEventCounter::value() const {
  if (is_finished()) {
    throw std::runtime_error(
        "Error in OccEventCounter::value: Counting is finished");
  }
  return m_data->occ_event;
}

bool OccEventCounter::is_finished() const { return m_stepper->is_finished(); }

}  // namespace occ_events
}  // namespace CASM
