import os 
import pickle

def save_stub(stub_path,object):
    """
    Save the stub data to a file.

    Args:
        stud_path (str): Path to the stub file.
        stub_data (any): Data to be saved in the stub file.
    """
    if not os.path.exists(os.path.dirname(stub_path)):
        os.mkdir(os.path.dirname(stub_path))

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object,f)


def read_stub(read_from_stub,stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            object=pickle.load(f)

    return object



    
    