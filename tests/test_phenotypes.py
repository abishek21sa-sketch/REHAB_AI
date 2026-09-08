import numpy as np
from rehab_ai.ml.phenotypes import RehabilitationPhenotyper


def test_phenotype_model_has_separated_synthetic_archetypes():
    m=RehabilitationPhenotyper(seed=5)
    silhouette=m.fit_validate(n_per=60)
    assert silhouette > 0.45
    r=m.predict(np.array([0.3,0.5,0.4,0.35,0.3]))
    assert r.phenotype in set(m.labels.values())
    assert r.validation_scope == 'SYNTHETIC VALIDATION'
