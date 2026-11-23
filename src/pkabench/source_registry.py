"""Source provenance and compatible-context ownership, independent of splitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .runtime import config_hash


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    first_party: bool
    data_kind: str
    native_split_field: str
    component_field: str
    input_scope_id: str
    adapter: str

    def __post_init__(self):
        if (not isinstance(self.first_party, bool) or
                any(not isinstance(v, str) or not v for k, v in asdict(self).items()
                    if k != 'first_party') or self.data_kind not in {'paired', 'site_only'}):
            raise ValueError('Invalid source specification')


@dataclass(frozen=True)
class SourceRegistry:
    sources: tuple[SourceSpec, ...]

    def __post_init__(self):
        if not self.sources or len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError('Registry source IDs must be unique')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if set(data) != {'schema_version', 'sources'} or data['schema_version'] != 1:
            raise ValueError('Unsupported source registry schema')
        return cls(tuple(SourceSpec(**s) for s in data['sources']))

    def to_dict(self):
        return {'schema_version': 1, 'sources': [asdict(s) for s in sorted(self.sources, key=lambda s: s.source_id)]}

    @property
    def sha256(self):
        return config_hash(self.to_dict())

    def get(self, source_id):
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise ValueError('Unregistered source: ' + str(source_id))

    def validate_precedence(self, precedence):
        required = {s.source_id for s in self.sources if s.first_party}
        if len(precedence) != len(required) or set(precedence) != required:
            raise ValueError('Precedence must list each first-party source exactly once')

    def choose_owner(self, records, precedence):
        self.validate_precedence(precedence)
        if not records:
            raise ValueError('No ownership candidates')
        for record in records:
            self.get(record['source'])
        for field in ('prepared_key', 'teacher_context_id'):
            if any(not r.get(field) for r in records) or len({r[field] for r in records}) != 1:
                raise ValueError('Ownership requires identical established ' + field)
        eligible = [r for r in records if r.get('eligible') is True]
        if not eligible or any(r.get('operational_role') not in {'train', 'validation', 'hard_test'} for r in eligible):
            raise ValueError('Ownership requires role-assigned eligible records')
        if len({r['operational_role'] for r in eligible}) != 1:
            raise ValueError('Equivalent records have conflicting roles; reassess before ownership')
        if any(r['source'] not in precedence for r in eligible):
            raise ValueError('Ownership source is outside first-party precedence')
        return min(eligible, key=lambda r: (precedence.index(r['source']), r['source_native_id']))
