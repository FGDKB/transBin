import numpy as np

def validate_input_array(array):
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    if not array.flags['OWNDATA']:
        array = array.copy()

    assert (array.flags['C_CONTIGUOUS'] and array.flags['OWNDATA'])
    return array

npz = np.load('latent.npz')
array = validate_input_array(npz['arr_0'])
print(array.shape)