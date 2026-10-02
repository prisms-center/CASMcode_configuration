#include "casm/configuration/occ_events/io/json/OccEventCounter_json_io.hh"

#include <optional>

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/configuration/clusterography/io/json/IntegralCluster_json_io.hh"
#include "casm/configuration/occ_events/OccEventCounter.hh"
#include "casm/configuration/occ_events/OccSystem.hh"
#include "casm/configuration/occ_events/io/json/OccEvent_json_io.hh"

namespace CASM {

jsonParser &to_json(occ_events::OccEventCounterStateInfo const &state_info,
                    jsonParser &json, occ_events::OccSystem const &system) {
  to_json(state_info.occ_init, json["occ_init"]);
  to_json(state_info.occ_final, json["occ_final"]);
  to_json(state_info.position_init, json["position_init"], system);
  to_json(state_info.position_final, json["position_final"], system);
  to_json(state_info.cluster, json["cluster"], *system.prim);
  to_json(state_info.occ_event, json["event"], system);
  if (state_info.fails.size()) {
    to_json(state_info.fails, json["fails"]);
  } else {
    to_json(true, json["passes"]);
  }
  return json;
}

namespace {

struct _ParamsToJson {
  _ParamsToJson(jsonParser &_json, bool _include_if_null = false,
                bool _include_if_default = false)
      : json(_json),
        include_if_null(_include_if_null),
        include_if_default(_include_if_default) {}

  jsonParser &json;
  bool include_if_null;
  bool include_if_default;

  template <typename T>
  void operator()(std::optional<T> const &t, std::string name) {
    if (include_if_null || t.has_value()) {
      to_json(t, json[name]);
    }
  }

  template <typename T>
  void as_array(std::optional<T> const &t, std::string name) {
    if (include_if_null || t.has_value()) {
      to_json(t, json[name], jsonParser::as_array());
    }
  }

  template <typename T>
  void if_not_default(T t, T default_value, std::string name) {
    if (include_if_default || t != default_value) {
      to_json(t, json[name]);
    }
  }
};

}  // namespace

jsonParser &to_json(occ_events::OccEventCounterParameters const &params,
                    jsonParser &json) {
  _ParamsToJson _to_json(json);
  //
  _to_json(params.min_cluster_size, "min_cluster_size");
  _to_json(params.max_cluster_size, "max_cluster_size");
  _to_json(params.required_cluster_size, "required_cluster_size");
  _to_json(params.excluded_sublattices, "excluded_sublattices");
  _to_json(params.required_sublattices, "required_sublattices");
  _to_json(params.n_required_sublattices, "n_required_sublattices");

  _to_json(params.required_occ_init, "required_occ_init");

  // Order determined by OccSystem::atom_name_list
  _to_json.as_array(params.min_init_atom_count, "min_init_atom_count");
  _to_json.as_array(params.max_init_atom_count, "max_init_atom_count");
  _to_json.as_array(params.required_init_atom_count,
                    "required_init_atom_count");

  // Order determined by OccSystem::molecule_name_list
  _to_json.as_array(params.min_init_molecule_count, "min_init_molecule_count");
  _to_json.as_array(params.max_init_molecule_count, "max_init_molecule_count");
  _to_json.as_array(params.required_init_molecule_count,
                    "required_init_molecule_count");

  // Order determined by OccSystem::orientation_name_list
  _to_json.as_array(params.min_init_orientation_count,
                    "min_init_orientation_count");
  _to_json.as_array(params.max_init_orientation_count,
                    "max_init_orientation_count");
  _to_json.as_array(params.required_init_orientation_count,
                    "required_init_orientation_count");

  _to_json.as_array(params.required_occ_final, "required_occ_final");

  // Order determined by OccSystem::atom_name_list
  _to_json.as_array(params.min_final_atom_count, "min_final_atom_count");
  _to_json.as_array(params.max_final_atom_count, "max_final_atom_count");
  _to_json.as_array(params.required_final_atom_count,
                    "required_final_atom_count");

  // Order determined by OccSystem::molecule_name_list
  _to_json.as_array(params.min_final_molecule_count,
                    "min_final_molecule_count");
  _to_json.as_array(params.max_final_molecule_count,
                    "max_final_molecule_count");
  _to_json.as_array(params.required_final_molecule_count,
                    "required_final_molecule_count");

  // Order determined by OccSystem::orientation_name_list
  _to_json.as_array(params.min_final_orientation_count,
                    "min_final_orientation_count");
  _to_json.as_array(params.max_final_orientation_count,
                    "max_final_orientation_count");
  _to_json.as_array(params.required_final_orientation_count,
                    "required_final_orientation_count");

  _to_json.if_not_default(params.allow_subcluster_events, false,
                          "allow_subcluster_events");
  _to_json.if_not_default(params.do_not_allow_breakup, false,
                          "do_not_allow_breakup");
  _to_json.if_not_default(params.skip_direct_exchange, true,
                          "skip_direct_exchange");
  _to_json.if_not_default(params.save_state_info, false, "save_state_info");
  return json;
}

void parse(InputParser<occ_events::OccEventCounterParameters> &parser) {
  parser.value = std::make_unique<occ_events::OccEventCounterParameters>();
  occ_events::OccEventCounterParameters &params = *parser.value;

  parser.optional(params.min_cluster_size, "min_cluster_size");
  parser.optional(params.max_cluster_size, "max_cluster_size");
  parser.optional(params.required_cluster_size, "required_cluster_size");
  parser.optional(params.excluded_sublattices, "excluded_sublattices");
  parser.optional(params.required_sublattices, "required_sublattices");
  parser.optional(params.n_required_sublattices, "n_required_sublattices");

  parser.optional(params.required_occ_init, "required_occ_init");

  // Order determined by OccSystem::atom_name_list
  parser.optional(params.min_init_atom_count, "min_init_atom_count");
  parser.optional(params.max_init_atom_count, "max_init_atom_count");
  parser.optional(params.required_init_atom_count, "required_init_atom_count");

  // Order determined by OccSystem::molecule_name_list
  parser.optional(params.min_init_molecule_count, "min_init_molecule_count");
  parser.optional(params.max_init_molecule_count, "max_init_molecule_count");
  parser.optional(params.required_init_molecule_count,
                  "required_init_molecule_count");

  // Order determined by OccSystem::orientation_name_list
  parser.optional(params.min_init_orientation_count,
                  "min_init_orientation_count");
  parser.optional(params.max_init_orientation_count,
                  "max_init_orientation_count");
  parser.optional(params.required_init_orientation_count,
                  "required_init_orientation_count");

  parser.optional(params.required_occ_final, "required_occ_final");

  // Order determined by OccSystem::atom_name_list
  parser.optional(params.min_final_atom_count, "min_final_atom_count");
  parser.optional(params.max_final_atom_count, "max_final_atom_count");
  parser.optional(params.required_final_atom_count,
                  "required_final_atom_count");

  // Order determined by OccSystem::molecule_name_list
  parser.optional(params.min_final_molecule_count, "min_final_molecule_count");
  parser.optional(params.max_final_molecule_count, "max_final_molecule_count");
  parser.optional(params.required_final_molecule_count,
                  "required_final_molecule_count");

  // Order determined by OccSystem::orientation_name_list
  parser.optional(params.min_final_orientation_count,
                  "min_final_orientation_count");
  parser.optional(params.max_final_orientation_count,
                  "max_final_orientation_count");
  parser.optional(params.required_final_orientation_count,
                  "required_final_orientation_count");

  parser.optional_else(params.allow_subcluster_events,
                       "allow_subcluster_events", false);
  parser.optional_else(params.do_not_allow_breakup, "do_not_allow_breakup",
                       false);
  parser.optional_else(params.skip_direct_exchange, "skip_direct_exchange",
                       true);
  parser.optional_else(params.save_state_info, "save_state_info", false);

  if (!parser.valid()) {
    parser.value.reset();
  }
}

}  // namespace CASM
