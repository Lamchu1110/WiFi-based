"""Named architecture controls; each comparison declares what changes."""
VARIANTS = {
    'baseline_legacy': ('baseline','legacy','standard','legacy'),
    'pooling_legacy': ('baseline','legacy','preserve_csi','legacy'),
    'tcn_legacy': ('tcn','legacy','preserve_csi','legacy'),
    'baseline_corrected': ('baseline','corrected','standard','legacy'),
    'pooling_corrected': ('baseline','corrected','preserve_csi','legacy'),
    'tcn_corrected': ('tcn','corrected','preserve_csi','legacy'),
    'tcn_gated': ('tcn','corrected','standard','short_gated'),
}
DEFAULT_VARIANTS = ['baseline_legacy','baseline_corrected','tcn_gated']
COMPARISONS = [
    ('baseline_legacy','baseline_corrected','DSK layout only'),
    ('baseline_legacy','pooling_legacy','pooling only, inherited DSK'),
    ('pooling_legacy','tcn_legacy','late TCN only, identical pooling and inherited DSK'),
    ('baseline_corrected','pooling_corrected','pooling only, corrected DSK'),
    ('pooling_corrected','tcn_corrected','late TCN only, identical pooling and corrected DSK'),
    ('baseline_corrected','tcn_gated','early short gated TCN only, identical DSK and pooling'),
]
