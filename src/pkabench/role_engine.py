"""Pure metadata role decisions. No old held-out cohort veto or source ranking.

Strict-profile evidence is deliberately unable to veto published train/validation
assignments. It is used separately for external hard-test feasibility.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace


PUBLISHED = 'published_abag_v0.3.0'
STRICT = 'strict_v1'


@dataclass(frozen=True)
class Unit:
    unit_id: str
    source: str
    native_split: str
    input_scope_id: str
    native_component: str = ''
    supervised: bool = True
    declared_general: bool = False
    required_axes: tuple[str, ...] = ('antibody_CDR', 'antigen')


@dataclass(frozen=True)
class Evidence:
    unit_id: str
    profile: str
    axis: str
    rule_id: str
    status: str
    components: tuple[str, ...] = ()

    def __post_init__(self):
        if self.profile not in {PUBLISHED, STRICT} or self.axis not in {'antibody_CDR', 'antigen', 'exact', 'interface', 'generic'}:
            raise ValueError('Unknown evidence profile or axis')
        if self.status not in {'hit', 'no_hit', 'unassessed'}:
            raise ValueError('Unknown evidence status')
        if (self.status == 'hit') != bool(self.components):
            raise ValueError('Only hits carry matched reference components')


@dataclass(frozen=True)
class Assignment:
    unit_id: str
    source: str
    native_source_split: str
    native_component: str
    input_scope_id: str
    operational_role: str
    reason_codes: tuple[str, ...]
    matched_components: tuple[str, ...]
    authority_sha256: str
    registry_sha256: str


def assign(units, evidence, component_splits, authority, registry):
    """Deterministically assign candidate roles from declared published-scope evidence.

An authoritative unit cannot move through inferred links. Additional links to
the opposite role instead quarantine that unit, preserving the component's
original role. Missing required checks keep any candidate pending.
"""
    registry.validate_precedence(authority.source_precedence)
    authority_sha, registry_sha = authority.sha256, registry.sha256
    authority_source = authority.to_dict()['authority']['source']
    units = list(units)
    ids = {u.unit_id for u in units}
    if len(ids) != len(units):
        raise ValueError('Duplicate unit IDs')
    components = {c: authority.role(s) for c, s in component_splits.items()}
    by_unit = defaultdict(list)
    for edge in evidence:
        if edge.unit_id not in ids:
            raise ValueError('Evidence for unknown unit')
        if edge.profile == PUBLISHED:
            if not set(edge.components) <= components.keys():
                raise ValueError('Evidence references unknown authoritative component')
            by_unit[edge.unit_id].append(edge)
    out = []
    for u in sorted(units, key=lambda u: u.unit_id):
        registry.get(u.source)
        if not u.input_scope_id:
            raise ValueError('Undeclared input scope')
        matches = {c for e in by_unit[u.unit_id] for c in e.components}
        own = u.source == authority_source
        if own:
            if components.get(u.native_component) != authority.role(u.native_split):
                raise ValueError('Authoritative component and split disagree')
            matches.add(u.native_component)
        missing = [axis for axis in u.required_axes if not any(
            e.axis == axis and e.status != 'unassessed' for e in by_unit[u.unit_id]) or any(
            e.axis == axis and e.status == 'unassessed' for e in by_unit[u.unit_id])]
        roles = {components[c] for c in matches}
        if not u.supervised:
            role, reasons = 'diagnostic', ('outside_supervised_cohort',)
        elif len(roles) > 1:
            role, reasons = 'quarantine', ('bridges_authoritative_train_validation',)
        elif missing:
            role, reasons = 'pending', tuple('unassessed:' + a for a in sorted(missing))
        elif roles:
            role, reasons = next(iter(roles)), ('authoritative_component' if own else 'inherits_published_components',)
        elif u.declared_general and u.native_split in {'train', 'validation'}:
            role, reasons = u.native_split, ('unmapped_declared_general_group_split',)
        else:
            role, reasons = 'pending', ('unmapped_antibody_or_unsupported_source_split',)
        out.append(Assignment(u.unit_id, u.source, u.native_split, u.native_component,
            u.input_scope_id, role, reasons, tuple(sorted(matches)), authority_sha, registry_sha))
    return out


def hard_test_status(checks, required_axes=('antibody_CDR', 'antigen', 'generic', 'interface'), prior_use='unknown'):
    """Strict checks must cover BOTH development roles on every required axis.

Known overlap disproves eligibility even when some other axes are unassessed.
Sequence-clean but structurally unassessed cases remain pending.
"""
    by_key = defaultdict(list)
    for axis, against_role, status in checks:
        if against_role not in {'train', 'validation'} or status not in {'hit', 'no_hit', 'unassessed'}:
            raise ValueError('Invalid hard-test evidence')
        by_key[(axis, against_role)].append(status)
    if any('hit' in values for values in by_key.values()):
        return 'diagnostic_overlap'
    if any(not by_key[(axis, role)] or 'unassessed' in by_key[(axis, role)]
           for axis in required_axes for role in ('train', 'validation')):
        return 'pending_novelty_checks'
    return 'candidate_not_frozen' if prior_use == 'never_used' else 'diagnostic_prior_use_unresolved'


def quarantine_group_conflicts(assignments, unit_groups, authoritative_source='sabdab', preferred_removal_source='pinder', *, explain=False):
    """Conservatively preserve declared family and exact-input group boundaries.

    Group IDs are namespaced by the caller (source families or cross-source
    exact input sequences). No ownership or duplicate claim follows from an
    exact-sequence group. Authoritative components keep their published roles;
    non-authoritative conflicting records are withheld pending preparation.
    """
    group_roles = defaultdict(set)
    group_members = defaultdict(list)
    for a in assignments:
        if a.operational_role in {'train', 'validation'}:
            for group in unit_groups[a.unit_id]:
                group_roles[group].add(a.operational_role)
                group_members[group].append(a)
    bad = {g for g, rr in group_roles.items() if len(rr) > 1}
    remove = set()
    decisions = []
    for group in sorted(bad):
        members = group_members[group]
        fixed = {a.operational_role for a in members if a.source == authoritative_source}
        if len(fixed) > 1:
            raise ValueError('Conflicting authoritative inputs; scope review required')
        if fixed:
            selected = {a.unit_id for a in members if a.operational_role not in fixed}
            rule = 'authoritative_anchor'
        else:
            rest = {a.operational_role for a in members if a.source != preferred_removal_source}
            if len(rest) <= 1:
                selected = {a.unit_id for a in members if a.source == preferred_removal_source}
                rule = 'preferred_source_removal'
            else:
                selected = {a.unit_id for a in members}
                rule = 'mixed_nonpreferred_roles'
        remove.update(selected)
        if explain:
            decisions.append(dict(group_id=group, rule=rule,
                fixed_anchor_roles=sorted(fixed),
                train_members=sorted(a.unit_id for a in members if a.operational_role == 'train'),
                validation_members=sorted(a.unit_id for a in members if a.operational_role == 'validation'),
                removed_members=sorted(selected)))
    out = []
    for a in assignments:
        if a.unit_id in remove:
            out.append(replace(a, operational_role='quarantine',
                reason_codes=a.reason_codes + ('mixed_family_or_exact_input_group',)))
        else:
            out.append(a)
    remaining = defaultdict(set)
    for a in out:
        if a.operational_role in {'train', 'validation'}:
            for g in unit_groups[a.unit_id]:
                remaining[g].add(a.operational_role)
    if any(len(rr) > 1 for rr in remaining.values()):
        raise ValueError('Conflicting authoritative inputs; scope review required')
    return (out, decisions) if explain else out
