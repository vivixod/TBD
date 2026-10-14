"""Training-only, class-first query sampling mixed with existing episode sampling."""
import numpy as np


class ClassBalancedQueryMixin:
    def set_class_balance_prob(self, probability):
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError('class_balance_prob must be between 0 and 1')
        # Validation/test must not change sampling or consume additional RNG draws.
        self.class_balance_prob = probability if self.split == 'trn' else 0.0
        self.class_query_pools = {}
        self.balanced_class_ids = ()
        if self.class_balance_prob == 0:
            return
        allowed_queries = set(self.img_metadata)
        for class_id in self.class_ids:
            # Network IDs are zero-based; classwise metadata uses pixel labels (ID+1).
            candidates = self.img_metadata_classwise.get(class_id + 1, ())
            pool = tuple(dict.fromkeys(name for name in candidates if name in allowed_queries))
            if pool:
                self.class_query_pools[int(class_id)] = pool
        self.balanced_class_ids = tuple(self.class_query_pools)
        if not self.balanced_class_ids:
            raise ValueError(f'{self.benchmark}: no training query pool for class-balanced sampling')

    def sample_class_balanced_query(self):
        probability = getattr(self, 'class_balance_prob', 0.0)
        if self.split != 'trn' or probability == 0:
            return None
        if probability < 1 and np.random.rand() >= probability:
            return None
        class_id = int(np.random.choice(self.balanced_class_ids))
        pool = self.class_query_pools[class_id]
        query_name = pool[int(np.random.randint(len(pool)))]
        return query_name, class_id


def configure_class_balance(dataset, probability):
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError('class_balance_prob must be between 0 and 1')
    if hasattr(dataset, 'set_class_balance_prob'):
        dataset.set_class_balance_prob(probability)
    elif probability > 0:
        raise ValueError(f'{dataset.benchmark}: class-balanced query sampling is unsupported')
