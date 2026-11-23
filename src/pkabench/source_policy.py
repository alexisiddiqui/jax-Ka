"""Versioned source ownership and evaluation protection for combined pools.

Source precedence is consulted only after quality/context compatibility. It is
not permission to train on a protected test or validation record. Loading a
different policy never modifies a previously released pool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

SOURCES = frozenset({'pkpdb', 'sabdab', 'pinder'})


@dataclass(frozen=True)
class SourcePolicy:
    schema_version: int
    policy_id: str
    source_precedence: tuple[str, ...]
    evaluation_policy: str
    validation_policy: str
    preparation_policy: str
    duplicate_policy: str
    sampling_policy: str
    apo_policy: str

    def __post_init__(self):
        if self.schema_version != 1 or not self.policy_id:
            raise ValueError('Unsupported or unnamed source policy')
        if len(self.source_precedence) != len(SOURCES) or set(self.source_precedence) != SOURCES:
            raise ValueError('Source precedence must contain pkpdb, sabdab and pinder exactly once')
        allowed = {
            'evaluation_policy': {'protect_official_sabdab_test', 'preserve_pkpdb_report_clean_test_subset'},
            'validation_policy': {'legacy_plus_published_components'},
            'preparation_policy': {'full_observed_source_chains'},
            'duplicate_policy': {'identical_prepared_context_only'},
            'sampling_policy': {'preserve_existing_min_fraction'},
            'apo_policy': {'existing_augmentation_no_new_pkai'},
        }
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError('Unsupported '+name+': '+str(getattr(self, name)))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        data['source_precedence'] = tuple(data['source_precedence'])
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        data['source_precedence'] = list(self.source_precedence)
        return data

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def rank(self, source):
        if source not in SOURCES:
            raise ValueError('Unknown source: '+str(source))
        return self.source_precedence.index(source)

    def choose_owner(self, records: Sequence[Mapping]):
        """Select provenance ownership only for proven-identical eligible units.

        A sequence, PDB accession or half-Fv lineage ID is not a prepared key.
        Unprepared inventory rows deliberately cannot enter this function.
        Teacher context is checked independently to fail closed on bad callers.
        """
        if not records:
            raise ValueError('No ownership candidates')
        if any(not r.get('prepared_key') or not r.get('teacher_context_id') for r in records):
            raise ValueError('Prepared identity and teacher context must be established before ownership')
        if len({r['prepared_key'] for r in records}) != 1:
            raise ValueError('Different prepared units must not be merged')
        if len({r['teacher_context_id'] for r in records}) != 1:
            raise ValueError('Teacher context mismatch')
        for r in records:
            self.rank(r['source'])
        eligible = [r for r in records if r.get('eligible') is True]
        if not eligible:
            raise ValueError('No quality-eligible ownership candidate')
        return min(eligible, key=lambda r: (self.rank(r['source']), str(r['source_native_id'])))

    def training_action(self, source, protected_role):
        """Return an audit decision; never mutate source membership.

        The alternative evaluation profile is explicit and must exclude the
        affected test component from clean evaluation. It is not the default.
        """
        self.rank(source)
        if protected_role not in {'external_heldout', 'sabdab_test', 'validation'}:
            raise ValueError('Unknown protection role: '+str(protected_role))
        if (source == 'pkpdb' and protected_role == 'sabdab_test' and
                self.evaluation_policy == 'preserve_pkpdb_report_clean_test_subset'):
            return 'keep_training_exclude_affected_test_component'
        return 'exclude_training'
