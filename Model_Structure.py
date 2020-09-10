import warnings, sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import keras
import numpy as np
from keras.utils import plot_model
from contextlib import redirect_stdout
np.set_printoptions(threshold=np.nan)

model = keras.models.load_model("Models/Generalized_Model.hdf5")
    
with open('Model_Summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        
plot_model(model, to_file='Model_Structure.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#from IPython.display import SVG 
#from keras.utils.vis_utils import model_to_dot 
#SVG(model_to_dot(model).create(prog='dot', format='svg'))