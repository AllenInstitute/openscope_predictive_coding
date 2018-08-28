"""
OpenScope OPhys Stimulus Script
"""
import os
from psychopy import visual
from camstim import Stimulus, SweepStim, MovieStim, NaturalScenes
from camstim import Foraging
from camstim import Window, Warp
import glob
import numpy as np
import hashlib
import json
import warnings


# data_path = r'//allen/aibs/technology/nicholasc/openscope'

warnings.warn('DEV PATH IN USE')
data_path = r'//allen/aibs/technology/nicholasc/openscope_dev'

expected_gray_screen_duration = 60.0
expected_randomized_control_duration = 105.0
expected_oddball_stimulus_list_duration = 2000.0
expected_pair_control_duration = 360.0
expected_occlusion_duration = 600.0
expected_familiar_movie_duration = 300.0
expected_total_duration = 3830.0
session_type = 'A'

hash_dict = {
            (68, 78, 13, 26): '2950e8d1e5187ce65ac40f5381be0b3f',
            (114, 23, 65, 101):'e17602dcefbd0e2ec66f81b46e4583f0',
            (53, 44, 8, 14):'0dbffd94a00dc4e9ea22ad75662d36bd',
            (68, 78, 13, 6):"e7c9e5b69add976b510ee8e85fc9a451",
            (68, 78, 13, 22):"eb69aeef83b4852217df8f8a2eb529c7",
            (68, 78, 13, 51):"b8b6a093ff7955ea8fd2cb48bd5ffa3c",
            (68, 78, 13, 71):"59880b3ff357998b5fac5b79b60c1e92",
            (68, 78, 13, 111):"a8465d0d71cace488afe32c2dc977eb5",
            (68, 78, 13, 17):"b4466dd964710092bbbce1a5c62acd38",
            (68, 78, 13, 110):"60f4700eaa8f6cb0ca07027482398e36",
            (68, 78, 13, 112):"8df3919368d45af726c7d3f57f88d9db",
            (68, 78, 13, 103):"03ef13047bedfa9a6981c190ff29eb1a",
            (68, 78, 13, 89):"0ba9d593323f104e447af570d8f1e9bf",
            (53, 44, 8, 7):"a778301437cd2db86d45d89002183137",
            (53, 44, 8, 94):"2bcd24821c058a5cfaeb0f3452c69b41",
            (53, 44, 8, 9):"ab543042e98a51b02354ffb9484e7753",
            (53, 44, 8, 29):"ecf045c6ab975986c6f34ccbba24d564",
            (53, 44, 8, 117):"ea89f286662939c77e6771bb266f7e81",
            (53, 44, 8, 42):"919f850548f20293952648cd8e8c1416",
            (53, 44, 8, 10):"afb21d3a3723e4d6d28cb8516506ed2e",
            (53, 44, 8, 45):"643889e712ede518ebf437f8b5ea607d",
            (53, 44, 8, 41):"7aa199407b52df9259c459f503314ff4",
            (53, 44, 8, 19):"5040e50ec95a95c18c9fa59e71c214b4",
            (114, 23, 65, 96):"cf19c50fdf456554c1eb23e691df72e0",
            (114, 23, 65, 84):"44d1a684c434044919e8a1f5625f47fc",
            (114, 23, 65, 32):"7957e0d36452766565eedd2dcdfe7c64",
            (114, 23, 65, 15):"6aad25b98dcfba07a2df440f220ca789",
            (114, 23, 65, 115):"10c7de386b8d4ec0bbdc22f94471cb86",
            (114, 23, 65, 27):"526942ab6cb1ff1a6da8036b60dc63a0",
            (114, 23, 65, 40):"af1e5e9a92a1ba7f0f27396f11505a54",
            (114, 23, 65, 52):"665f3b91f9f86e890193a6e3b719c9b3",
            (114, 23, 65, 93):"a57ca8e9a7cd21227be001134ba486b7",
            (114, 23, 65, 35):"a45a4af032d77d51f8070b806929d0e5",
            (68, 78):"5eab8e4d385e03923d533d07d3e45584",
            (78, 13):"6a67e1eccc79a46306d159749bcc8269",
            (13, 26):"45e997f20a2d56bc54a31b2ef3c4d2a6",
            (26, 68):"56949723a23eb5cf490f11ca4da31aef",
            (13, 6):"b689a86c3aed3544ce7038a5164ce8d0",
            (6, 68):"753d1dc8898f90cfddc0a34260c99b94",
            (13, 22):"a6c62110e211048c57e421cf5f783bac",
            (22, 68):"e90bc7a3204e763dc7e295f87473dd38",
            (13, 51):"c4ff749440c735e2441bdf19e32f3347",
            (51, 68):"7ce633b022206ef58e26a0a89baaf72c",
            (13, 71):"4c979c7b2cfc6be45d8111a161bcd181",
            (71, 68):"e87911f3940b74e261f1f1b05e5ea467",
            (13, 111):"d59bd95cd944f4a948bd983aa3393c5e",
            (111, 68):"af3cde68447d1e6e5ec715c7acbd62ab",
            (13, 17):"c19ee95e08a25696fc5e7c2edfd20f94",
            (17, 68):"c44244b2818c1d9d8305fbd84cebea08",
            (13, 110):"9fb8a433dfff62b6fbc061c99cf64682",
            (110, 68):"ed871f890173b657723d0a1e83289a0d",
            (13, 112):"75ee28af87f82a069754803a2c1e92cf",
            (112, 68):"4fffefd365c0de14aee4f506f7a8d4ef",
            (13, 103):"b9a33c3b401538ae053144a78ca918ec",
            (103, 68):"7fcb9723dc41d69987662fe9c9fc6fd6",
            (13, 89):"10a592bc1d0136eac9c332023c4fad1a",
            (89, 68):"b052d2f81769e9ccdc6d19ff2f53a02f",
            (53, 44):"3f41f43ad02d6ebee0fed87217f8b27b",
            (44, 8):"2c7066de6bc94802cae6eb66de281f4a",
            (8, 14):"257c56d360067a7fe46977aa3e9e6dff",
            (14, 53):"6a4a39092ade258e081fcf9df21cd424",
            (8, 7):"1e90f71f52a3055d5702b1cee8655bb3",
            (7, 53):"cc9716b1a5301633e02345b78efb3152",
            (8, 94):"56f529a115b860a9ea2668aaf553d045",
            (94, 53):"bb0c5922ca3e7826521349bf266e431f",
            (8, 9):"a5bcd193dec7f157f9dc01526d6a1d0b",
            (9, 53):"98cbcb57ad39ef00faeb43beea771202",
            (8, 29):"67cb0b931280888f5abe40343d3ad81f",
            (29, 53):"381e1769cf667e6db9386306aedfe009",
            (8, 117):"531db73f293c90845990a730da35287f",
            (117, 53):"b15f5c166c3ab8c48e4edd17a00ee36e",
            (8, 42):"721c3bb6653508b3190ecb178f4d1968",
            (42, 53):"888b6d1ba0827d9fc68adc7f09ce5b7c",
            (8, 10):"fa53a4ee8edb2e85019c089afb841518",
            (10, 53):"d1b98d43a31ec62f56bdae0c981fd541",
            (8, 45):"a4cffb0e78ee67a98a982f3d251db1c6",
            (45, 53):"69583c41bfd28d059f4c90344584a898",
            (8, 41):"e8d062e3d90f0a30af57014c92701bbc",
            (41, 53):"43aa8bf617e5f4d52cf60cea5f9bc833",
            (8, 19):"d07a3e13814744bfa5ca555f6eb9ad85",
            (19, 53):"e1b2531095df59bb53bb23379e305074",
            (114, 23):"658da57b1f22b0d301105dbf07a45a10",
            (23, 65):"093a62463e2f36ae637369df9d8ad31b",
            (65, 101):"d43023a45c5adf9d9cfdfe89ca3a5706",
            (101, 114):"71eec9c3891355482ccee53ae2b04a86",
            (65, 96):"74c3b91b31d1c40e38997bc861d85394",
            (96, 114):"d13b7bdebd59f710520b0f9a5a918f71",
            (65, 84):"83be8e162ea916db3cb248d73ac2c5c8",
            (84, 114):"f7cbfc935c110fec24c76799bb5f9a41",
            (65, 32):"9cb0a0ed6840009dc9491145bed4e3bb",
            (32, 114):"d1ddd0de04a99f82ea16cca2285ab84a",
            (65, 15):"b301d2c2d237dfb468c634b15ed2143e",
            (15, 114):"fa548dfc1705e7373917dbec6260d6d0",
            (65, 115):"5c61d5f679bd79f9d31f30984d232dba",
            (115, 114):"ed6a486234e2100236deab4f4b558f45",
            (65, 27):"799d15bed31c4ce4dfd22f96666e3e46",
            (27, 114):"d0e0439788af5e04f82a0331cece20b8",
            (65, 40):"9d72954d8611d2d3ef42e36cd065f1c0",
            (40, 114):"186f0838b66fcc4d8a564e8e1c6f6018",
            (65, 52):"239ff6e0f980f229e11e8096fd91e930",
            (52, 114):"051a0c07fa8ed2b3f23715b856c9c16c",
            (65, 93):"19ebf98db451eff81a395d0515c15d17",
            (93, 114):"79abb0826ba10f70ba2d95116c20b5d7",
            (65, 35):"882ac80abe0350d5d1b9d79f97b08c31",
            (35, 114):"a42192da281a9bbeca2b5debbdcc5f07",
            }

# Consistency check:
assert os.path.basename(__file__).split('.')[0][-1] == session_type

# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0)

interval_data = []
def get_block(file_name, timing_list, frame_length, runs, t0):
    
    base_seq_stim = MovieStim(movie_path=file_name,
                                    window=window,
                                    frame_length=frame_length,
                                    size=(1920, 1200),
                                    start_time=0.0,
                                    stop_time=None,
                                    flip_v=True,
                                    runs=runs,)

    # Shift t0:
    timing_hwm = -float('inf')
    timing_list_new = []
    for t_start, t_end in timing_list:
        t_start_new = t_start+t0
        t_end_new = t_end+t0
        timing_list_new.append((t_start_new, t_end_new))
        interval_data.append(((t_start_new, t_end_new), file_name, runs))
        if t_end_new > timing_hwm:
            timing_hwm = t_end_new

    base_seq_stim.set_display_sequence(timing_list_new)
    return base_seq_stim, timing_hwm

# Initialize:
tf = 0
stimuli = []

# Spontaneous gray screen block 1:
tf += 60

# Randomized oddball block:
t0 = tf
file_name = os.path.join(data_path, 'ophys_pilot_randomized_control_%s_598ac1255d9c8e09541ae1f57034fac3.npy' % session_type)
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 1
frame_length = .25
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_randomized_control_duration

# Spontaneous gray screen block 1:
tf += 60

# oddball_stim_list sequence block:
t0 = tf
oddball_list = json.load(open(os.path.join(data_path, 'oddball_timing_data_%s.json' % session_type), 'r'))
oddball_stimulus_list = []
tf_list = []
frame_length = .25
for pattern, timing_list in oddball_list:
    file_name = os.path.join(data_path, '%s_%s.npy' % ('_'.join([str(x) for x in pattern]), hash_dict[tuple(pattern)]))
    curr_stim, curr_tf = get_block(file_name, timing_list, frame_length, len(timing_list), t0=t0, )
    oddball_stimulus_list.append(curr_stim)
    tf_list.append(curr_tf)
tf = max(tf_list)
stimuli += oddball_stimulus_list
assert tf - t0 == expected_oddball_stimulus_list_duration

# Spontaneous gray screen block 2:
tf += 60

# Pair control sequence block:
t0 = tf
pair_list = json.load(open(os.path.join(data_path, 'transition_pair_timing_data_%s.json' % session_type), 'r'))
pair_stimulus_list = []
tf_list = []
frame_length = .25
for pattern, timing_list in pair_list:
    file_name = os.path.join(data_path, '%s_%s.npy' % ('_'.join([str(x) for x in pattern]), hash_dict[tuple(pattern)]))
    curr_stim, curr_tf = get_block(file_name, timing_list, frame_length, len(timing_list), t0=t0, )
    pair_stimulus_list.append(curr_stim)
    tf_list.append(curr_tf)
tf = max(tf_list)
stimuli += pair_stimulus_list
assert tf - t0 == expected_pair_control_duration

# Spontaneous gray screen block 3:
tf += 60

# occlusion movie block:
t0 = tf
file_name = os.path.join(data_path, 'ophys_pilot_occlusion_659b61b30d270bb849a7c4a903125e02.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 1
frame_length = .5
timing_list = [(.5+ii, (ii+1)) for ii in range(number_of_frames)] 
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0)
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_occlusion_duration

# Spontaneous gray screen block 4:
tf += 60

# Familiar movie block:
t0 = tf
file_name = os.path.join(data_path, 'natural_movie_one_warped_77ee4ecd0dc856c80cf24621303dd080.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 10
frame_length = 2.0/60.0
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0)
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_familiar_movie_duration

# Spontaneous gray screen block 5:
tf += 60

# Randomized oddball block:
t0 = tf
file_name = os.path.join(data_path, 'ophys_pilot_randomized_control_A_598ac1255d9c8e09541ae1f57034fac3.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 1
frame_length = .25
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_randomized_control_duration

for ii in interval_data:
    print ii

sys.exit()

assert tf == expected_total_duration
params = {}
ss = SweepStim(window,
            stimuli=stimuli,
            pre_blank_sec=0,
            post_blank_sec=0,
            params=params,
            )

f = Foraging(window=window,
            auto_update=False,
            params=params,
            nidaq_tasks={'digital_input': ss.di,
                        'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")
ss.run()
