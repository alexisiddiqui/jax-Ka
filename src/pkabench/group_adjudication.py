"""Inactive counterfactuals for legacy groups under fixed published authority.

All group removals are explainable. These policies produce proposals; they do
not alter source membership, prepared context, teacher labels or sampling.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace, asdict

from .runtime import config_hash


ACTIVE = frozenset({'train', 'validation'})


@dataclass(frozen=True)
class GroupPolicy:
    policy_id: str
    mapped_family_mode: str = 'retain'
    anchor_propagation: bool = False
    antigen_reference_scope: str = 'resolved'
    preferred_removal_source: str = 'pinder'

    def __post_init__(self):
        if (not self.policy_id or self.mapped_family_mode not in {'retain', 'published_components'} or
                type(self.anchor_propagation) is not bool or
                self.antigen_reference_scope not in {'resolved', 'resolved_expected_union'} or
                not self.preferred_removal_source):
            raise ValueError('Invalid inactive group policy')
        if self.anchor_propagation and self.mapped_family_mode != 'published_components':
            raise ValueError('Anchor propagation requires explicit mapped-family authority')

    @property
    def sha256(self):
        return config_hash(asdict(self))


def make_groups(units, members, assignments, *, exempt_mapped=False):
    assignments = {a.unit_id: a for a in assignments}
    out = {uid: set() for uid in units}
    for uid, u in units.items():
        a = assignments[uid]
        if u['source'] == 'sabdab' or (exempt_mapped and a.matched_components and a.operational_role in ACTIVE):
            continue
        out[uid].update('family:' + u['source'] + ':' + g for g in u['source_groups'])
    for m in members:
        out[m['unit_id']].add('exact:' + m['sequence_key'])
    return out


def propagate_family_anchors(assignments, units):
    """Propagate only to assessed, unmapped GENERAL records with agreeing anchors.

    Matched units never move. Unmapped antibodies and unassessed records remain
    pending. Opposing anchors cannot be joined through a general-protein unit.
    """
    anchors = defaultdict(set)
    for a in assignments:
        if a.source == 'sabdab' or not a.matched_components or a.operational_role not in ACTIVE:
            continue
        for g in units[a.unit_id]['source_groups']:
            anchors[(a.source, g)].add(a.operational_role)
    out, log = [], []
    for a in assignments:
        if a.operational_role not in ACTIVE or a.matched_components or not units[a.unit_id]['declared_general']:
            out.append(a)
            continue
        evidence = [(g, anchors[(a.source, g)]) for g in units[a.unit_id]['source_groups'] if anchors[(a.source, g)]]
        desired = set().union(*(rr for _, rr in evidence)) if evidence else set()
        if len(desired) > 1:
            new = replace(a, operational_role='quarantine', reason_codes=a.reason_codes + ('conflicting_authoritative_family_anchors',))
        elif desired:
            new = replace(a, operational_role=next(iter(desired)), reason_codes=a.reason_codes + ('family_role_from_authoritative_anchor',))
        else:
            new = a
        out.append(new)
        if evidence:
            log.append(dict(unit_id=a.unit_id, previous_role=a.operational_role,
                proposed_role=new.operational_role, desired_roles=sorted(desired),
                family_groups=sorted(g for g, _ in evidence)))
    return out, log


def resolve_groups(assignments, unit_groups, *, preferred_removal_source='pinder', authoritative_source='sabdab'):
    """Resolve anchored and removable-source constraints before broad removals.

    This is a deterministic conservative refinement, not a globally minimal
    graph cut. Re-evaluation prevents an already removed unit from causing a
    later removal. Within a priority tier, smaller groups precede larger groups.
    """
    aa = {a.unit_id: a for a in assignments}
    group_members = defaultdict(set)
    for uid, groups in unit_groups.items():
        for g in groups:
            group_members[g].add(uid)
    # Removing records cannot create a mixed group. Avoid rescanning the
    # complete pool for every decision in the small conflicting subgraph.
    group_members = {g: uu for g, uu in group_members.items()
                    if len({aa[u].operational_role for u in uu if aa[u].operational_role in ACTIVE}) > 1}
    log = []
    while True:
        choices = []
        for group in sorted(group_members):
            members = [aa[u] for u in sorted(group_members[group]) if aa[u].operational_role in ACTIVE]
            roles = {a.operational_role for a in members}
            if len(roles) < 2:
                continue
            fixed = {a.operational_role for a in members if a.source == authoritative_source}
            if len(fixed) > 1:
                raise ValueError('Conflicting authoritative inputs; scope review required')
            if fixed:
                selected = {a.unit_id for a in members if a.operational_role not in fixed}
                tier, rule = 0, 'authoritative_anchor'
            else:
                rest = {a.operational_role for a in members if a.source != preferred_removal_source}
                if len(rest) == 1:
                    selected = {a.unit_id for a in members if a.source == preferred_removal_source}
                    tier, rule = 1, 'preferred_source_removal'
                else:
                    selected = {a.unit_id for a in members}
                    tier, rule = 2, 'remaining_mixed_group'
            choices.append(((tier, 0 if group.startswith('exact:') else 1, len(members), group),
                selected, members, rule, fixed))
        if not choices:
            break
        key, selected, members, rule, fixed = min(choices, key=lambda c: c[0])
        group = key[-1]
        log.append(dict(step=len(log), group_id=group, rule=rule,
            fixed_anchor_roles=sorted(fixed),
            train_members=sorted(a.unit_id for a in members if a.operational_role == 'train'),
            validation_members=sorted(a.unit_id for a in members if a.operational_role == 'validation'),
            removed_members=sorted(selected)))
        for uid in selected:
            a = aa[uid]
            aa[uid] = replace(a, operational_role='quarantine',
                reason_codes=a.reason_codes + ('adjudicated_group_conflict',))
    return [aa[a.unit_id] for a in assignments], log


def assert_boundaries(assignments, unit_groups):
    roles = defaultdict(set)
    for a in assignments:
        if a.operational_role in ACTIVE:
            for g in unit_groups[a.unit_id]:
                roles[g].add(a.operational_role)
    bad = {g: rr for g, rr in roles.items() if len(rr) > 1}
    if bad:
        raise ValueError('Retained group crosses roles: ' + str(sorted(bad)[:5]))
