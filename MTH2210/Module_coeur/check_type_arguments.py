#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%##################################
# Import des bibliothèques requises #
#####################################

import types
import numpy as np



#%%######################################
# Définition des fonctions élémentaires #
#########################################

# Renvoie le type de l'argument arg, sous forme d'une chaîne de caractères de la forme "<class 'type_de_arg'>"
#   get_type("123")             = "<class 'str'>"
#   get_type(lambda x:x)        = "<class 'function'>"
#   get_type(np.array([1,2,3])) = "<class 'numpy.ndarray'>"
def get_type(arg):
    return(str(type(arg)))

# Vérifie la concordance entre le type de arg et le type attendu. Renvoie le booléen et le vrai type de arg.
#   check_fundamental("123", str)                     = True,  "<class 'str'>"
#   check_fundamental("123", float)                   = False, "<class 'str'>"
#   check_fundamental(100.0, int)                     = False, "<class 'float'>"
#   check_fundamental(100.0, float)                   = True,  "<class 'float'>"
#   check_fundamental(lambda x:x, types.FunctionType) = True,  "<class 'function'>"
#   check_fundamental(get_type, types.FunctionType)   = True,  "<class 'function'>"
#   check_fundamental(np.array([1,2,3]), np.ndarray)  = True,  "<class 'numpy.ndarray'>"
def check_fundamental(arg, expected_type):
    return(isinstance(arg, expected_type), get_type(arg))



#%%##########################################
# Définition des fonctions pour chaque type #
#############################################

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est une fonction); et (type est le type de arg)
def check_function(arg):
    return(check_fundamental(arg, types.FunctionType))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est une chaîne de caractères); et (type est le type de arg)
def check_str(arg):
    return(check_fundamental(arg, str))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est un int, float, int ou np.float64); et (type est le type de arg)
def check_real(arg):
    if check_fundamental(arg, int)[0] == True:
        return(True, get_type(arg))
    elif check_fundamental(arg, np.float64)[0] == True:
        return(True, get_type(arg))
    elif check_fundamental(arg, float)[0] == True:
        return(True, get_type(arg))
    elif check_fundamental(arg, int)[0] == True:
        return(True, get_type(arg))
    else:
        return(False, get_type(arg))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est un int ou int); et (type est le type de arg)
def check_int(arg):
    return(check_fundamental(arg, int))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est un np.ndarray); et (type est le type de arg)
def check_nparray(arg):
    return(check_fundamental(arg, np.ndarray))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est une liste); et (type est le type de arg)
def check_list(arg):
    return(check_fundamental(arg, list))

# Renvoie un couple (bool, type) où (bool = True si et seulement si arg est du type expected_type); et (type est le type de arg)
def check_generic(arg, expected_type):
    if expected_type == types.FunctionType:
        return(check_function(arg))
    if expected_type == str:
        return(check_str(arg))
    if expected_type == np.float64:
        return(check_real(arg))
    if expected_type == int:
        return(check_int(arg))
    if expected_type == np.ndarray:
        return(check_nparray(arg))
    if expected_type == list:
        return(check_list(arg))



#%%###################################################
# Définition de la fonction appelée par les méthodes #
######################################################

# Appelle la fonction de vérification de chacun des paramètres contenus dans le tableau reçu.
# args_list doit avoir la structure suivante :
#   args_list = [ [param_1, nom_param_1, type_attendu_param_1],
#                 [param_2, nom_param_2, type_attendu_param_2],
#                 ...,
#                 [param_N, nom_param_N, type_attendu_param_N] ]
#   où pour tout i,
#       param_i est le paramètre,
#       nom_param_i une chaîne de caractères,
#       type_attendu_param_i est le type que param_i doit avoir, ou une liste des types que param_i peut avoir.
# La fonction ne renvoie rien mais signale une ValueError si au moins un des paramètres n'est pas du type attendu.
def check_parameters(args_list):
    len_n = 0
    len_c = 0
    for _, name, expected in args_list:
        len_n = max(len_n, len(name))
        len_c = max(len_c, len(str(expected)))
    buffer_errors = []
    for arg, name, expected in args_list:
        if type(expected) == list:
            is_ok, given = check_generic(arg, expected[0])
            for exp in expected:
                is_ok_exp, given_exp = check_generic(arg, exp)
                if is_ok_exp:
                    is_ok = True
                    given = given_exp
        else:
            is_ok, given = check_generic(arg, expected)
        if not(is_ok):
            s = "Paramètre {:{len_n}} : attendu {:<{len_c}} , reçu {}".format(name, str(expected), str(given), len_n=len_n, len_c=len_c)
            buffer_errors.append(s)
    if buffer_errors != []:
        buffer_errors = "\n".join(["Problèmes de type des paramètres :"]+buffer_errors)
        raise ValueError(buffer_errors)
