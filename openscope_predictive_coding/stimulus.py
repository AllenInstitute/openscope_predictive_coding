import numpy as np
import os
import hashlib
import json

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import get_hash, apply_warp_natural_scene, apply_warp_natural_movie, luminance_match, downsample_monitor_to_template, get_shuffled_repeated_sequence, generate_sequence_block, seq_to_str, get_shuffled_repeated_sequence, generate_sequence_block, generate_oddball_block_timing_dict, generate_pair_block_timing_dict, memoized
from occlusion import get_occlusion_data_metadata

default_manifest = os.path.join(opc.boc_path, 'boc/manifest.json')
default_data_path = opc.data_path


TEMPLATE_LIST = [si.NATURAL_MOVIE_ONE, si.NATURAL_MOVIE_TWO, si.NATURAL_SCENES]
TEMPLATE_LIST_WARPED = ["%s_warped" % s for s in TEMPLATE_LIST]
NATURAL_SCENES_WARPED = si.NATURAL_SCENES + '_warped'
HABITUATION_PILOT_RANDOMIZED_ODDBALL = 'habituation_pilot_randomized_oddball'
HABITUATION_PILOT_DATA = [HABITUATION_PILOT_RANDOMIZED_ODDBALL]

stimulus_oeid_dict = {
                     si.NATURAL_MOVIE_ONE: 511458874,
                     si.NATURAL_SCENES: 511458874,
                     si.NATURAL_MOVIE_TWO: 527550473
                     }

hash_dict = {
            si.NATURAL_MOVIE_ONE: "b174ad09736c870c6915baf82cf2c9ad",
            si.NATURAL_MOVIE_TWO: "68e5976a140fe8400c6b7fe59073fe72",
            si.NATURAL_SCENES: "b9a9a5284200f80b56ba6f4eecd34712",
            si.NATURAL_MOVIE_TWO + '_warped': '92f1fe36e2c118761cbebcebcc6cd076',
            si.NATURAL_MOVIE_ONE + '_warped': '77ee4ecd0dc856c80cf24621303dd080',
            NATURAL_SCENES_WARPED: '8ba4262b06ec81c3ec8d3d7d7831e564',
            HABITUATION_PILOT_RANDOMIZED_ODDBALL: '5cd9854e9cb07a427180d6e130c148ab',
            opc.SEQUENCE_IMAGES['A']: '2950e8d1e5187ce65ac40f5381be0b3f',
            opc.SEQUENCE_IMAGES["C"]:'e17602dcefbd0e2ec66f81b46e4583f0',
            opc.SEQUENCE_IMAGES["B"]:'0dbffd94a00dc4e9ea22ad75662d36bd',
            'ophys_pilot_randomized_control_A':"598ac1255d9c8e09541ae1f57034fac3",
            'ophys_pilot_randomized_control_B':"1ce104b1011311ac984e647054fd253f",
            'ophys_pilot_randomized_control_C':"9864cb4ff0140082826f46608ebeb6cc",
            'ophys_pilot_occlusion':"8e3b57fa469782eb610ba2bfad2c4f37",
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


STIMULUS_LIST = TEMPLATE_LIST+TEMPLATE_LIST_WARPED+HABITUATION_PILOT_DATA

assert set(stimulus_oeid_dict.keys()) == set(TEMPLATE_LIST)
assert NATURAL_SCENES_WARPED in TEMPLATE_LIST_WARPED

def get_stimulus_path(stimulus_key, data_path=default_data_path, append_hash=True):
    if isinstance(stimulus_key, tuple):
        stimulus = stimulus_key
        stimulus_key = seq_to_str(stimulus_key)
    else:
        stimulus = stimulus_key

    if append_hash:
        return os.path.join(data_path, '%s_%s.npy' % (stimulus_key, hash_dict[stimulus]))
    else:
        return os.path.join(data_path, '%s.npy' % (stimulus_key, ))

def remove_warped_from_stimulus_key(stimulus_name):
    assert stimulus_name in STIMULUS_LIST

    split_stimulus = stimulus_name.split('_')
    assert split_stimulus[-1] == 'warped'
    return '_'.join(split_stimulus[:-1])
    
@memoized
def get_stimulus_template_brain_observatory(stimulus, data_path=default_data_path, manifest_file=default_manifest):
    
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    dataset_path = get_stimulus_path(stimulus, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
        oeid = stimulus_oeid_dict[stimulus]
        data_set = boc.get_ophys_experiment_data(oeid)
        data = data_set.get_stimulus_template(stimulus)
        assert hash_dict[stimulus] == get_hash(data)
        np.save(dataset_path, data)

    assert hash_dict[stimulus] == get_hash(data)
    return data
    
@memoized
def get_stimulus_template_warped(stimulus_key, data_path=default_data_path, manifest_file=default_manifest):
    
    stimulus_key_prewarp = remove_warped_from_stimulus_key(stimulus_key)

    data = get_stimulus_template_brain_observatory(stimulus_key_prewarp, data_path, manifest_file)
    data_warp_path = get_stimulus_path(stimulus_key, data_path=data_path)
    if os.path.exists(data_warp_path):
        data_warp = np.load(data_warp_path)
    else:
        data_warp = np.empty((data.shape[0], opc.IMAGE_H, opc.IMAGE_W), dtype=np.uint8)
        for fi, img in enumerate(data):
            if stimulus_key_prewarp in si.NATURAL_MOVIE_STIMULUS_TYPES:
                img_warp = apply_warp_natural_movie(img)
                img_warp_ds = downsample_monitor_to_template(img_warp)
                data_warp[fi,:,:] = img_warp_ds
            elif stimulus_key_prewarp == si.NATURAL_SCENES:
                img_warp = apply_warp_natural_scene(img)
                img_warp_lm = luminance_match(img_warp)
                img_warp_lm_ds = downsample_monitor_to_template(img_warp_lm)[::-1,:]
                data_warp[fi,:,:] = img_warp_lm_ds
            else:
                raise RuntimeError
            print stimulus_key, fi
        assert hash_dict[stimulus_key] == get_hash(data_warp)
        np.save(data_warp_path, data_warp)
    
    assert hash_dict[stimulus_key] == get_hash(data_warp)
    return data_warp

@memoized
def get_stimulus_randomized_oddball(**kwargs):
    data_path = kwargs.get('data_path', default_data_path)

    dataset_path = get_stimulus_path(HABITUATION_PILOT_RANDOMIZED_ODDBALL, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
        habituation_oddball_full_sequence = get_shuffled_repeated_sequence(opc.HABITUATED_ODDBALL_IMAGES, 100, seed=0)
        src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
        data = generate_sequence_block(habituation_oddball_full_sequence, src_image_data)
        assert hash_dict[HABITUATION_PILOT_RANDOMIZED_ODDBALL] == get_hash(data)
        np.save(dataset_path, data)

    assert hash_dict[HABITUATION_PILOT_RANDOMIZED_ODDBALL] == get_hash(data)
    return data

@memoized
def get_sequence_data_hash(sequence, **kwargs):
    src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED, **kwargs)
    data = generate_sequence_block(sequence, src_image_data)
    return data, get_hash(data)

@memoized
def get_sequence_template(sequence, **kwargs):
    data_path = kwargs.get('data_path', default_data_path)
    
    dataset_path = get_stimulus_path(sequence, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
        data, data_hash = get_sequence_data_hash(sequence, **kwargs)
        assert hash_dict[sequence] == data_hash
        np.save(dataset_path, data)
    
    assert hash_dict[sequence] == get_hash(data)
    return data


    

@memoized
def get_stimulus_template(stimulus, **kwargs):
    
    if stimulus in TEMPLATE_LIST:
        data = get_stimulus_template_brain_observatory(stimulus, **kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif stimulus in TEMPLATE_LIST_WARPED:
        data = get_stimulus_template_warped(stimulus, **kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif stimulus == HABITUATION_PILOT_RANDOMIZED_ODDBALL:
        data = get_stimulus_randomized_oddball(**kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif isinstance(stimulus, tuple):
        sequence = stimulus
        data = get_sequence_template(sequence, **kwargs)
    else:
        raise RuntimeError
            
    return data

def get_pilot_randomized_control(session, data_path=default_data_path):
    
    session_key = 'ophys_pilot_randomized_control_%s' % session
    data_hash = hash_dict[session_key]
    dataset_path = get_stimulus_path(session_key, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:

        ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session]
        SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session]
        pilot_randomized_control_full_sequence = get_shuffled_repeated_sequence(ODDBALL_IMAGES + SEQUENCE_IMAGES, 30, seed=1)
        
        src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
        data = generate_sequence_block(pilot_randomized_control_full_sequence, src_image_data)

        assert data_hash == get_hash(data)
        np.save(dataset_path, data)
    
    assert data_hash == get_hash(data)
    return data


def get_oddball_data(session, data_path=default_data_path):

    ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session]
    SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session]
    
    oddball_timing_dict = generate_oddball_block_timing_dict(SEQUENCE_IMAGES, ODDBALL_IMAGES, expected_duration=2000.0, seed=0)
    
    oddball_data_dict = {}
    for oddball_id in ODDBALL_IMAGES:
        tmp_seq = [x for x in SEQUENCE_IMAGES]
        tmp_seq[-1] = oddball_id
        tmp_seq = tuple(tmp_seq)

        dataset_path = get_stimulus_path(tmp_seq, data_path=data_path)
        data_hash = hash_dict[tmp_seq]

        if os.path.exists(dataset_path):
            data = np.load(dataset_path)
        else:

            src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
            data = generate_sequence_block(tmp_seq, src_image_data)

            assert data_hash == get_hash(data)
            np.save(dataset_path, data)
        
        assert data_hash == get_hash(data)
        oddball_data_dict[tmp_seq] = data
       
    return oddball_data_dict, oddball_timing_dict

    
def save_oddball_timing_dict(session, data_path=default_data_path):
    
    _, oddball_timing_dict = get_oddball_data(session)
    oddball_list = []
    for key_val in oddball_timing_dict.items():
        oddball_list.append(key_val)        
    
    save_file_name = os.path.join(data_path, 'oddball_timing_data_%s.json' % session)
    json.dump(oddball_list, open(save_file_name, 'w'))

def get_transition_pair_data(session, data_path=default_data_path):

    ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session]
    SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session]

    # Generate transition pair sequences:
    pair_list = []
    for ii in range(len(SEQUENCE_IMAGES)):
        jj = ii+1
        if jj >= len(SEQUENCE_IMAGES):
            jj = 0
        pair_list.append((SEQUENCE_IMAGES[ii], SEQUENCE_IMAGES[jj]))
    for oddball_image in ODDBALL_IMAGES:
        pair_list.append((SEQUENCE_IMAGES[-2], oddball_image))
        pair_list.append((oddball_image, SEQUENCE_IMAGES[0]))

    pair_data_dict = {}
    for pair in pair_list:

        dataset_path = get_stimulus_path(pair, data_path=data_path)
        data_hash = hash_dict[pair]

        if os.path.exists(dataset_path):
            data = np.load(dataset_path)
        else:
            
            src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
            data = generate_sequence_block(pair, src_image_data)

            assert data_hash == get_hash(data)
            np.save(dataset_path, data)
        
        assert data_hash == get_hash(data)
        pair_data_dict[pair] = data

    pair_timing_dict = generate_pair_block_timing_dict(pair_list, num_repeats=30, frame_length=.25, expected_duration=360., seed=0)

    return pair_data_dict, pair_timing_dict

def save_transition_pair_timing_dict(session, data_path=default_data_path):
    
    _, transition_pair_timing_dict = get_transition_pair_data(session)
    transition_pair_list = []
    for key_val in transition_pair_timing_dict.items():
        transition_pair_list.append(key_val)        
    
    save_file_name = os.path.join(data_path, 'transition_pair_timing_data_%s.json' % session)
    json.dump(transition_pair_list, open(save_file_name, 'w'))

def get_occlusion_data(data_path=default_data_path):

    stimulus_key = 'ophys_pilot_occlusion'
    dataset_path = get_stimulus_path(stimulus_key, data_path=data_path)
    data_hash = hash_dict[stimulus_key]

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:

        src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
        data, _ =  get_occlusion_data_metadata(opc.ODDBALL_IMAGES['A'], src_image_data)
        assert data_hash == get_hash(data)
        np.save(dataset_path, data)
        
    assert data_hash == get_hash(data)
    
    return data


if __name__ == "__main__":



    # for stimulus in STIMULUS_LIST:
    # template = get_stimulus_template(si.NATURAL_SCENES+'_warped')
    # for key, val in opc.SEQUENCE_IMAGES.items():
        # print 'opc.SEQUENCE_IMAGES["%s"]:"%s",' % (key, get_sequence_hash(val))
        # print get_stimulus_template(val)
    # template = 

    # get_stimulus_template(si.NATURAL_SCENES + '_warped')
    # save_oddball_timing_dict('A')
    # save_oddball_timing_dict('B')
    # save_oddball_timing_dict('C')
    # get_oddball_data('B')
    # get_oddball_data('C')
    # get_oddball_data('A')
    # get_transition_pair_data('A')
    # get_transition_pair_data('B')
    # get_transition_pair_data('C')
    # save_transition_pair_timing_dict('A')
    # save_transition_pair_timing_dict('B')
    # save_transition_pair_timing_dict('C')

    get_occlusion_data()

