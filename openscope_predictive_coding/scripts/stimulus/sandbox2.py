
from openscope_predictive_coding.utilities import run_camstim_debug
import openscope_predictive_coding as opc
import matplotlib.pyplot as plt

img_stack = opc.get_dataset_template('NATURAL_SCENES_OCCLUSION_WARPED')

plt.imshow(img_stack[0,:,:])
plt.show()

