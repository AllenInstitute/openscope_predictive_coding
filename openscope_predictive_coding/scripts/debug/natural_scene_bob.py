"""
three_session_B.py
"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0,
                warp=Warp.Spherical,)

# Paths for stimulus files
# sg_path = r"C:\Users\svc_flex4\Desktop\cam\cam2p_scripts\cam_1_0\static_gratings.stim"
# nm1_path = r"C:\Users\svc_flex4\Desktop\cam\cam2p_scripts\cam_1_0\natural_movie_1.stim"
ns_path = r"//allen/aibs/technology/nicholasc/openscope/natural_scenes.stim"

# Create stimuli
ns = Stimulus.from_file(ns_path, window)
# sg = Stimulus.from_file(sg_path, window)
# nm1 = Stimulus.from_file(nm1_path, window)

# set display sequences
ns_ds = [(510-510, 990-510)]
# sg_ds = [(0, 480), (1800, 2280), (3210, 3750)]
# nm1_ds = [(2310, 2610) ]

ns.set_display_sequence(ns_ds)
# sg.set_display_sequence(sg_ds)
# nm1.set_display_sequence(nm1_ds)

# kwargs
params = {
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [5, 6],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[ns],
               pre_blank_sec=20,
               post_blank_sec=2,
               params=params,
               )

# add in foraging so we can track wheel, potentially give rewards, etc
f = Foraging(window=window,
             auto_update=False,
             params=params,
             nidaq_tasks={'digital_input': ss.di,
                          'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")

# run it
ss.run()