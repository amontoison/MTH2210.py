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
    if not(f(x0)*f(x1) < 0):
        raise ValueError("Condition initiale f(x0)*f(x1) < 0 non respectée")
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
def format_iter(k, x_g, x_d, x_c, f_g, f_d, f_c):
    if k == 0:
        iter_infos  = "   k ||     x_g     |     x_d     ||     f_g     |     f_d     ||     x_c     |     f_c\n"
        iter_infos += "-------------------------------------------------------------------------------------------\n"
    else:
        iter_infos = ""
    iter_infos += "{:>4} || {:^+.4e} | {:^+.4e} || {:^+.4e} | {:^+.4e} || {:^+.4e} | {:^+.4e}".format(k, x_g, x_d, f_g, f_d, x_c, f_c)
    return(iter_infos)



#%%########################################
# Fonctions de tests des critères d'arrêt #
###########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(k, list_x, list_f, nb_iter, tol_abs, tol_rel):
    if k > nb_iter:
        return(True, "Nombre maximal d'itérations k_max={} autorisé dépassé".format(nb_iter))
    if abs(list_f[-1]) < tol_abs:
        return(True, "Racine localisée à {:2.1e} près : x = {:+8.7e} et f(x) = {:+8.7e}".format(tol_abs, list_x[-1], list_f[-1]))
    if k >= 1:
        err_rel_x = check_relative_tolerance.tol_rel_approx(list_x[-1], list_x[-2])
        if err_rel_x < tol_rel:
            return(True, "Convergence achevée à {:2.1e} près : x = {:+8.7e} et erreur relative sur x = {:5.4e}".format(tol_rel, list_x[-1], err_rel_x))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(f, x0, x1):
    k = 0
    x_g = min(x0, x1)
    x_d = max(x0, x1)
    x_c = (x_g+x_d)/2
    f_g = f(x_g)
    f_d = f(x_d)
    f_c = f(x_c)
    list_x = [x_c]
    list_f = [f_c]
    return(k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f)

# Exécute une itération de la méthode
def iter_algo(f, k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f):
    k += 1
    if f_g*f_c < 0:
        x_d = x_c
        x_c = (x_g+x_d)/2
        f_d = f_c
        f_c = f(x_c)
    elif f_c*f_d < 0:
        x_g = x_c
        x_c = (x_g+x_d)/2
        f_g = f_c
        f_c = f(x_c)
    list_x.append(x_c)
    list_f.append(f_c)
    return(k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f)



#%%#####################################
# Définition de la fonction principale #
########################################

def bissection(f, x0, x1, nb_iter=100, tol_rel=10**-8, tol_abs=10**-8, output=""):
    """Méthode de recherche d'une racine de la fonction f via la méthode de la bissection :
        - x_g et x_d donnés, vérifiant f(x_g)*f(x_d) < 0, et x_c = (x_g+x_d)/2 et f(x_c),
        - si f(x_c) = 0, on a trouvé la racine,
        - si f(x_g)*f(x_c) < 0, alors la méthode est relancée avec x_g inchangé et x_d = x_c,
        - si f(x_c)*f(x_d) < 0, alors la méthode est relancée avec x_g = x_c et x_d inchangé,
        - à chaque itération k, le point noté x_k est x_c.
    
    Les arguments attendus sont :
        - une fonction f, admettant en entrée un scalaire x et renvoyant un scalaire f(x),
        - deux scalaires x0 et x1 (de type int, float ou np.float64), les bornes de l'intervalle de recherche contenant la racine à localiser.
    
    Les arguments optionnels sont :
        - un entier nb_iter défiinissant le nombre maximal d'itérations allouées à la méthode,
        - un réel tol_rel définissant la condition d'arrêt abs(x_k-x_km1) / (abs(x_k)+eps) <= tol_rel,
        - un réel tol_abs définissant la condition d'arrêt abs(f(x_k)) <= tol_abs,
        - une chaîne de caractères output qui renvoie les affichages de la fonction vers :
            - la sortie standard si output = "",
            - un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
            - nul part (aucune information écrite ni sauvegardée) si output = "None".

    La méthode vérifie les conditions suivantes :
        - les bornes initiales doivent satisfaire f(x0)*f(x1) < 0 pour garantir l'existence d'une racine dans [x0,x1],
        - f est définie en x0 et x1, et renvoie en chacun de ces points un scalaire,
        - nb_iter, tol_rel et tol_abs sont positifs,
        - tous les paramètres reçus ont bien le type attendu.
    
    Les sorties de la méthode sont :
        - list_x, la liste des points centraux de l'intervalle de recherche à chaque itération (donc les approximations x_k de la racine),
        - list_f, les valeurs par f des éléments de list_x.
        
    Exemples d'appel :
        - bissection(lambda x : np.sin(x), -0.5, 1/3),
        - bissection(lambda x : 10**20*np.sin(x), -0.5, 0.25),
        - bissection(f, x0, x1, output="dossier_test/Résultats.txt") où f est définie via def, x0 et x1 sont deux réels.
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    check_parameters_consistency(f, x0, x1, nb_iter, tol_rel, tol_abs, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)
    
    # Initialisation de l'algorithme
    k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f = init_algo(f, x0, x1)
    write_iter(k, x_g, x_d, x_c, f_g, f_d, f_c)
    
    # Déroulement de l'algorithme
    while not(stopping_criteria(k, list_x, list_f, nb_iter, tol_abs, tol_rel)[0]):
        k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f = iter_algo(f, k, x_g, x_d, x_c, f_g, f_d, f_c, list_x, list_f)
        write_iter(k, x_g, x_d, x_c, f_g, f_d, f_c)
    
    write_stopping(stopping_criteria(k, list_x, list_f, nb_iter, tol_abs, tol_rel)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, list_f)


