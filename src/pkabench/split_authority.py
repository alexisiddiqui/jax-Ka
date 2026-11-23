"""Inactive, versioned split authority, separate from prepared-record ownership."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .runtime import config_hash


@dataclass(frozen=True)
class SplitAuthority:
    # Store canonical JSON rather than a mutable nested dict in a frozen class.
    canonical_json: str

    @classmethod
    def from_dict(cls, data):
        expected = {
            'schema_version', 'policy_id', 'status', 'authority', 'source_precedence',
            'precedence_scope', 'cross_source_assignment', 'legacy_cohorts',
            'hard_test', 'augmentation', 'representation', 'historical_experiments',
            'next_audit',
        }
        if set(data) != expected or data['schema_version'] != 2:
            raise ValueError('Unsupported or incomplete split authority schema')
        # This parser implements one policy, not arbitrary executable policy text.
        fixed = {
            'authority': {
                'source': 'sabdab', 'release': 'SAbDab2 ML v0.3.0',
                'partition_field': 'ab_ag_split', 'component_field': 'ab_ag_cluster',
                'operational_roles': {'train': 'train', 'test': 'validation'},
                'preserve_original_source_labels': True,
                'cohort': 'retained_protein_or_peptide_antigen',
            },
            'precedence_scope': 'eligible_proven_equivalent_prepared_units_not_split_authority',
            'cross_source_assignment': {
                'map_using': 'published_antibody_and_antigen_components_at_declared_input_scope',
                'training_component_only': 'train', 'validation_component_only': 'validation',
                'bridges_training_and_validation': 'quarantine_or_reprepare_nonbridging_units',
                'unmapped_general_proteins': 'retain_declared_family_group_split_after_holdout_checks',
                'incomplete_mapping': 'pending_not_assumed_clean',
                'prefer_removing_conflicting_source': 'pinder',
            },
            'legacy_cohorts': {
                'use': 'hard_test_candidates_or_labelled_diagnostics',
                'automatic_training_veto': False,
                'independent_experimental_claims': 'only_for_explicit_references_verified_clean_under_final_roles',
            },
            'augmentation': {'geometry_role_inherits': 'native_unit_role',
                             'new_apo_pkai': False, 'prevent_heldout_donors_in_training': True},
            'representation': {'scope_must_be_declared': True,
                               'changed_scope_requires_teacher_and_feature_compatibility_check': True,
                               'same_parent_PDB_proves_duplicate': False},
            'historical_experiments': {'retain_actual_memberships_and_split_versions': True,
                'retroactive_independence_claims': False,
                'checkpoint_exposure_requires_actual_run_membership': True},
            'hard_test': {
                'default_candidate_source': 'external_to_authoritative_training_and_validation',
                'novelty_profile': 'versioned_stricter_sequence_CDR_antigen_and_interface_rules',
                'check_against': ['train', 'validation'], 'unassessed_is_eligible': False,
                'freeze_before': 'model_selection_or_performance_inspection',
                'minimum_size_requires_feasibility_audit': True, 'relax_rules_to_fill_quota': False,
                'overlapping_legacy_cases': 'nonindependent_diagnostics',
                'carve_from_sabdab_published_test': 'requires_component_removal_from_validation_before_freeze',
            },
        }
        for key, value in fixed.items():
            if data[key] != value:
                raise ValueError('Unsupported split authority setting: ' + key)
        precedence = data['source_precedence']
        if (not isinstance(precedence, list) or not precedence or
                any(not isinstance(s, str) or not s for s in precedence) or
                len(set(precedence)) != len(precedence)):
            raise ValueError('Source precedence must contain distinct source IDs')
        if (not isinstance(data['policy_id'], str) or not data['policy_id'] or
                data['status'] != 'agreed_design_not_activated' or
                data['next_audit'] != 'reassign_proposed_pools_and_measure_hard_test_feasibility_before_native_builds'):
            raise ValueError('This engine supports inactive metadata audits only')
        return cls(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return json.loads(self.canonical_json)

    @property
    def sha256(self):
        return config_hash(self.to_dict())

    @property
    def source_precedence(self):
        return tuple(self.to_dict()['source_precedence'])

    def role(self, native_split):
        try:
            return self.to_dict()['authority']['operational_roles'][native_split]
        except KeyError as exc:
            raise ValueError('Unknown authoritative split: ' + str(native_split)) from exc
