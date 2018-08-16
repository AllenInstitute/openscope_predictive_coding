from openscope_predictive_coding.stimulus import get_brain_observatory_templates
from openscope_predictive_coding.utilities import get_hash


def test_get_brain_observatory_templates():
    
    hash_dict = {
                "natural_movie_two": "68e5976a140fe8400c6b7fe59073fe72",
                "natural_movie_one": "b174ad09736c870c6915baf82cf2c9ad",
                "natural_scenes": "b9a9a5284200f80b56ba6f4eecd34712"
                }

    template_dict = get_brain_observatory_templates()
    assert len(template_dict) == 3

    for key, val in template_dict.items():
        assert hash_dict[key] == get_hash(val)



# D = generate_oddball_block_timing_dict([0,1,2], [10,20], num_cycles_per_repeat=3, oddball_cycle_min=2, oddball_cycle_max=3, num_repeats_per_oddball=2, frame_length=.1, t0=.01, seed=0)
# D = generate_oddball_block_timing_dict([0,1,2], [10,20, 30, 40], num_cycles_per_repeat=3, oddball_cycle_min=2, oddball_cycle_max=3, num_repeats_per_oddball=2, frame_length=.001, t0=.01, seed=0)
# D = generate_oddball_block_timing_dict([0,1,2], [10,20, 30], num_cycles_per_repeat=3, oddball_cycle_min=2, oddball_cycle_max=3, num_repeats_per_oddball=10, frame_length=.15, t0=.01, seed=0)
# D = generate_oddball_block_timing_dict([0,1,2,3], [10,20, 30], num_cycles_per_repeat=2, oddball_cycle_min=2, oddball_cycle_max=2, num_repeats_per_oddball=2, frame_length=.1, t0=.01, seed=0)