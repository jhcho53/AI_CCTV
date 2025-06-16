import random
from torch.utils.data import DataLoader, Sampler

class BalancedBatchSampler(Sampler):
    """
    매 배치마다 정상/이상 샘플을 반반씩 뽑는 Sampler
    """
    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0, 
        self.batch_size  = batch_size
        self.half        = batch_size // 2
        self.pos_indices = [i for i, l in enumerate(labels) if l == 1.0]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0.0]
        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.pos_indices) >= self.half:
                pos = random.sample(self.pos_indices, self.half)
            else:
                pos = random.choices(self.pos_indices, k=self.half)
            if len(self.neg_indices) >= self.half:
                neg = random.sample(self.neg_indices, self.half)
            else:
                neg = random.choices(self.neg_indices, k=self.half)
            batch = pos + neg
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches