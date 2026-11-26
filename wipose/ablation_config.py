"""Named architecture controls; each comparison declares what changes."""
VARIANTS = {
    'baseline_legacy': ('baseline','legacy','standard','legacy'),
    'pooling_legacy': ('baseline','legacy','preserve_csi','legacy'),
    'tcn_legacy': ('tcn','legacy','preserve_csi','legacy'),
    'baseline_corrected': ('baseline','corrected','standard','legacy'),
    'pooling_corrected': ('baseline','corrected','preserve_csi','legacy'),
    'tcn_corrected': ('tcn','corrected','preserve_csi','legacy'),
    'tcn_gated': ('tcn','corrected','standard','short_gated'),
    'tcn_residual_legacy': ('tcn','legacy','standard','short_residual'),
    'tcn_residual_corrected': ('tcn','corrected','standard','short_residual'),
}
DEFAULT_VARIANTS = ['baseline_legacy','baseline_corrected','tcn_gated',
                    'tcn_residual_legacy','tcn_residual_corrected']
COMPARISONS = [
    ('baseline_legacy','baseline_corrected','DSK layout only'),
    ('baseline_legacy','pooling_legacy','pooling only, inherited DSK'),
    ('pooling_legacy','tcn_legacy','late TCN only, identical pooling and inherited DSK'),
    ('baseline_corrected','pooling_corrected','pooling only, corrected DSK'),
    ('pooling_corrected','tcn_corrected','late TCN only, identical pooling and corrected DSK'),
    ('baseline_corrected','tcn_gated','early short gated TCN only, identical DSK and pooling'),
    ('baseline_legacy','tcn_residual_legacy','early residual TCN only, original DSK and pooling'),
    ('baseline_corrected','tcn_residual_corrected','early residual TCN only, corrected DSK and identical pooling'),
    ('tcn_residual_legacy','tcn_residual_corrected','DSK layout only, identical residual TCN'),
    ('tcn_gated','tcn_residual_corrected','zero scalar gate versus zero output projection, identical corrected DSK'),
    ('baseline_legacy','tcn_residual_corrected','main original-vs-ours comparison: DSK and residual TCN together'),
]
