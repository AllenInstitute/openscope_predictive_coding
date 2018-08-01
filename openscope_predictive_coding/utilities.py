import random

def get_shuffled_repeated_sequence(src_sequence, number_of_shuffles, seed=None):
    
    if seed is not None:
        random.seed(seed)
    
    new_sequence = []
    for x in range(number_of_shuffles):
        random.shuffle(src_sequence)
        new_sequence += list(src_sequence)

    return new_sequence
