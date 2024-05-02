import numpy as np

def convert_opls_to_fourrier( opls: List[float], kcal_to_kj: bool=True ):

    C0 = opls[1] + 0.5*(opls[0]+opls[2])
    C1 = 0.5*(-opls[0] +3*opls[2])
    C2 = -opls[1] + 4*opls[3]
    C3 = -2*opls[2]
    C4 = -4*opls[3]
    C5 = 0
    
    convert_factor = 4.184 if kcal_to_kj else 1

    fourrier = np.array([C0, C1, C2, C3, C4, C5]) * convert_factor
    
    return fourrier


################## write topology to ff function  ##################