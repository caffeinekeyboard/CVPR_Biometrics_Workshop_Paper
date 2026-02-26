import numpy as np
from typing import Tuple

class TestDataLogic1:
    def __init__(self):
        self.idx_per_set = (8*7) // 2
        self.next_set = False
        self.running_max = 8
        self.running_id = 1
        self.data = np.arange(16)
    
    def get_item(self, idx: int) -> Tuple[int, int]:
        current_set = idx // self.idx_per_set
        if idx == current_set * self.idx_per_set:
            self.next_set = True
            print(f"Next set triggered at idx {idx} (current_set: {current_set})")
        if self.next_set:
            self.running_id = 1
            self.running_max = 8
            self.next_set = False
        if self.running_id == self.running_max:
            print(f"Running max reached at idx {idx} (current_set: {current_set}, running_id: {self.running_id})")
            self.running_id = 1
            self.running_max -= 1
        diff = 8 - self.running_max
        template_idx = (diff + self.running_id + current_set * 8)
        self.running_id += 1

        impression_idx = current_set * 8 + 8 - self.running_max

        return impression_idx, template_idx

class TestDataLogic2:
    def __init__(self):
        self.num_images = 10*8
        self.paths = []
        self.data = np.arange(16)
        self.pair_indices = []
        for idx in range(self.num_images * 8):
            impression_idx = idx // 8
            next_set = idx // 64
            template_idx = idx - impression_idx * 8 + next_set * 8

            self.pair_indices.append((impression_idx, template_idx))
    

    def get_item(self, idx: int) -> Tuple[int, int]:
        return self.pair_indices[idx]

def testlogic1():
    test_logic = TestDataLogic1()
    matrix = np.zeros((16, 16), dtype=int)
    for idx in range(56):
        impression, template = test_logic.get_item(idx)
        matrix[impression, template] = idx + 1
    print(matrix)

def testlogic2():
    test_logic = TestDataLogic2()
    list = []
    for idx in range(10 * 8 * 8):
        list.append(test_logic.get_item(idx))
    print(list[120:150])

if __name__ == "__main__":
    testlogic2()