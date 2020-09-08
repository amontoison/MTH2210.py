#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%##################################
# Import des bibliothèques requises #
#####################################

import numpy as np

#%%###################################################
# Définition de la fonction appelée par les méthodes #
######################################################

# Définit la fonction tol_rel_approx, qui calcule abs(elt1-elt2) / (abs(elt1)+epsilon_machine)
def tol_rel_approx(elt1, elt2):
    num = np.linalg.norm(elt1-elt2)
    den = np.linalg.norm(elt1) + np.spacing(1)
    return(num / den)


