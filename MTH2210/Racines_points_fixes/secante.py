#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%##################################
# Import des bibliothèques requises #
#####################################

from ..Module_coeur import check_type_arguments, check_relative_tolerance, writing_function
import types
import numpy as np



#%%########################################
# Fonction de vérification des paramètres #
###########################################

# Vérifie que le jeu de paramètres reçu par la méthode respecte les types attendus et les hypothèses mathématiques
def check_parameters_consistency(f, x0, x1, nb_iter, tol_rel, tol_abs, output):
    # Vérification des types des paramètres reçus
    params_array = [[f,       "f",       types.FunctionType],
                    [x0,      "x0",      np.float64],
                    [x1,      "x1",      np.float64],
                    [nb_iter, "nb_iter", np.int],
                    [tol_rel, "tol_rel", np.float64],
                    [tol_abs, "tol_abs", np.float64],
                    [output,  "output",  str]]
    check_type_arguments.check_parameters(params_array)
    # Vérification de la cohérence des paramètres
    try:
        f(x0)
    except:
        raise ValueError("Fonction f non définie en x0")
    try:
        f(x1)
    except:
        raise ValueError("Fonction f non définie en x1")
    if not(check_type_arguments.check_generic(f(x0), np.float64)[0]):
        raise ValueError("f(x0) n'est pas un scalaire (type reçu :"+check_type_arguments.get_type(f(x0))+")")
    if not(check_type_arguments.check_generic(f(x1), np.float64)[0]):
        raise ValueError("f(x1) n'est pas un scalaire (type reçu :"+check_type_arguments.get_type(f(x1))+")")
    if nb_iter < 0:
        raise ValueError("Condition d'arrêt nb_iter définie à une valeur négative")
    if tol_rel < 0:
        raise ValueError("Condition d'arrêt tol_rel définie à une valeur négative")
    if tol_abs < 0:
        raise ValueError("Condition d'arrêt tol_abs définie à une valeur négative")



#%%########################################
# Fonctions de mise en page des résultats #
###########################################

# Crée la chaîne de caractères qui sera renvoyée pour chaque itération
def format_iter(k, list_x, list_f, list_d):
    if k == 1:
        header  = "{:>4} || {:^11} | {:^11} | {:^11}"
        header  = header.format("k", "x_k", "f_k", "df(x_k)")
        header += "\n"
        header += "-"*(4+11+11+11 + 4+3+3)
        header += "\n"
        iter_infos_0 = "{:>4} || {:^+11.4e} | {:^+11.4e} | {:^11}".format("0", list_x[0], list_f[0], "/")
        iter_infos_0 += "\n"
        header += iter_infos_0
    else:
        header = ""
        iter_infos_0 = ""
    iter_infos = "{:>4} || {:>+11.4e} | {:>+11.4e} | {:>+11.4e}".format(k, list_x[-1], list_f[-1], list_d[-1])
    return(header+iter_infos)



#%%#####################################
# Fonctions de test du critère d'arrêt #
########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs):
    if k >= nb_iter:
        return(True, "Nombre maximal d'itérations k_max={} autorisé dépassé".format(nb_iter))
    if abs(list_f[-1]) < tol_abs:
        return(True, "Racine localisée à {:7.1e} près : x_k = {:+14.7e} et f(x_k) = {:+11.4e}".format(tol_abs, list_x[-1], list_f[-1]))
    if list_d[-1] == 0:
        return(True, "Dérivée exactement nulle au point courant x_k = {:+11.4e}".format(list_x[-1]))
    if k > 1:
        err_rel = check_relative_tolerance.tol_rel_approx(list_x[-1], list_x[-2])
        if err_rel < tol_rel:
            return(True, "Convergence de la méthode achevée à {:7.1e} près : df/dx(x_k) = {:+11.4e}".format(tol_rel, err_rel))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(f, x0, x1):
    k = 1
    x_km1 = x0
    x_k = x1
    f_km1 = f(x_km1)
    f_k = f(x_k)
    d_k = (f_k-f_km1)/(x_k-x_km1)
    list_x = [x_km1, x_k]
    list_f = [f_km1, f_k]
    list_d = [d_k]
    return(k, list_x, list_f, list_d)

# Exécute une itération de la méthode
def iter_algo(f, k, list_x, list_f, list_d):
    x_km1, x_k = list_x[-2:]
    f_km1, f_k = list_f[-2:]
    d_k = (f_k-f_km1)/(x_k-x_km1)
    k += 1
    x_k -= f_k/d_k
    list_x.append(x_k)
    list_f.append(f(x_k))
    list_d.append(d_k)
    return(k, list_x, list_f, list_d)



#%%#####################################
# Définition de la fonction principale #
########################################

def secante(f, x0, x1, nb_iter=100, tol_rel=10**-8, tol_abs=10**-8, output=""):
    """Méthode de recherche d'une racine de la fonction scalaire f via la méthode de la sécante :
        - x_0 et x_1 donnés,
        - d_k = (f(x_k)-f(x_km1)) / (x_k-x_km1),
        - x_kp1 = x_k - f(x_k) / d_k.
    
    Les arguments attendus sont :
        - une fonction f, admettant en entrée un scalaire x et renvoyant un scalaire f(x),
        - deux scalaires x0 et x1 (de type int, float ou np.float64), points de départ de la méthode itérative.
    
    Les arguments optionnels sont :
        - un entier nb_iter (défaut = 100 ) défiinissant le nombre maximal d'itérations allouées à la méthode,
        - un réel   tol_rel (défaut = 1e-8) définissant la condition d'arrêt abs(x_k-x_km1) / (abs(x_k)+eps) <= tol_rel,
        - un réel   tol_abs (défaut = 1e-8) définissant la condition d'arrêt abs(f(x_k)) <= tol_abs,
        - une chaîne de caractères output (défaut = "") qui renvoie les affichages de la fonction vers :
            - la sortie standard si output = "pipe",
            - un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
            - nul part (aucune information écrite ni sauvegardée) si output = "" ou output = "None".

    La méthode vérifie les conditions suivantes :
        - f est définie en x0 et x1, et renvoie un scalaire,
        - df est définie en x0 et x1, et renvoie un scalaire,
        - tous les paramètres reçus ont bien le type attendu.
    
    Les sorties de la méthode sont :
        - list_x, la liste des points x_k,
        - list_f, les valeurs par f des éléments de list_x,
        - list_d, la liste des approximations d_k des dérivées de f en les x_k.
        
    Exemples d'appel :
        - secante(lambda x : np.sin(x), 1, 0.5),
        - secante(lambda x :x**2, 2, 1).
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    check_parameters_consistency(f, x0, x1, nb_iter, tol_rel, tol_abs, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)
    
    # Initialisation de l'algorithme
    k, list_x, list_f, list_d = init_algo(f, x0, x1)
    write_iter(k, list_x, list_f, list_d)
    
    # Déroulement de l'algorithme
    while not(stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs)[0]):
        k, list_x, list_f, list_d = iter_algo(f, k, list_x, list_f, list_d)
        write_iter(k, list_x, list_f, list_d)
    
    write_stopping(stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, list_f, list_d)


