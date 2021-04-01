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
def check_parameters_consistency(f, jac, x0, nb_iter, tol_rel, tol_abs, output):
    # Vérification des types des paramètres reçus
    params_array = [[f,       "f",       types.FunctionType],
                    [jac,     "jac",     types.FunctionType],
                    [x0,      "x0",      np.ndarray],
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
        jac(x0)
    except:
        raise ValueError("Fonction Jac(f) non définie en x0")
    if not(check_type_arguments.check_generic(f(x0), np.ndarray)[0]):
        raise ValueError("f(x0) n'est pas un vecteur (type reçu :"+check_type_arguments.get_type(f(x0))+")")
    if not(check_type_arguments.check_generic(jac(x0), np.ndarray)[0] or check_type_arguments.check_generic(jac(x0), np.matrix)[0]):
        raise ValueError("Jac(f)(x0) n'est pas une matrice (type reçu :"+check_type_arguments.get_type(jac(x0))+")")
    if not(np.size(f(x0)) == np.size(x0)):
        raise ValueError("f(x0) n'a pas la même dimension que x0 (dimension reçue : "+str(np.size(f(x0)))+" et attendue : "+str(np.size(x0))+")")
    if not(np.shape(jac(x0)) == (np.size(x0),np.size(x0))):
        raise ValueError("Jac(f)(x0) n'a pas la même dimension que x0*x0 (dimension reçue : "+str(np.shape(jac(x0)))+" et attendue : "+str((np.size(x0),np.size(x0)))+")")
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
def format_iter(k, list_x, list_f):
    x_k = list_x[-1]
    f_k = list_f[-1]
    if k == 0:
        n = len(x_k)
        len_str_xk = 2+11*n+2*(n-1)
        header  = "{:>4} || " + "{:^"+str(len_str_xk)+"}" + " | " + "{:^"+str(len_str_xk)+"}"
        header  = header.format("k", "x_k", "f(x_k)")
        header += "\n"
        header += "-"*(4+len_str_xk+len_str_xk + 4+3)
        header += "\n"
    else:
        header  = ""
    iter_infos  = "{:>4} || ".format(k)
    iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in x_k])+"] | "
    iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in f_k])+"]"
    return(header+iter_infos)



#%%#####################################
# Fonctions de test du critère d'arrêt #
########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs):
    if k >= nb_iter:
        return(True, "Nombre maximal d'itérations k_max={} autorisé dépassé".format(nb_iter))
    if np.max(np.abs(list_f[-1])) < tol_abs:
        return(True, "Racine localisée à {:7.1e} près : x = ".format(tol_abs)+ "["+", ".join(["{:>+11.4e}".format(xi) for xi in list_x[-1]])+"]" +" et f(x) = "+"["+", ".join(["{:>+11.4e}".format(xi) for xi in list_f[-1]])+"]")
    if np.linalg.det(list_d[-1]) == 0:
        return(True, "Jacobienne singulière au point courant x_k = "+"["+", ".join(["{:>+11.4e}".format(xi) for xi in list_x[-1]])+"]")
    if k > 1:
        err_rel = check_relative_tolerance.tol_rel_approx(list_x[-1], list_x[-2])
        if err_rel < tol_rel:
            return(True, "Convergence de la méthode achevée à {:7.1e} près : df/dx(x_k) = {:+11.4e}".format(tol_rel, err_rel))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(f, jac, x0):
    k = 0
    x_k = x0
    f_k = f(x_k)
    d_k = jac(x_k)
    list_x = [x_k]
    list_f = [f_k]
    list_d = [d_k]
    return(k, list_x, list_f, list_d)

# Exécute une itération de la méthode
def iter_algo(f, jac, k, list_x, list_f, list_d):
    x_k = list_x[-1]
    f_k = list_f[-1]
    d_k = list_d[-1]
    k += 1
    x_k = x_k - np.linalg.solve(d_k,f_k)
    list_x.append(x_k)
    list_f.append(f(x_k))
    list_d.append(jac(x_k))
    return(k, list_x, list_f, list_d)



#%%#####################################
# Définition de la fonction principale #
########################################

def newton_nd_avec_der(f, jac, x0, nb_iter=100, tol_rel=10**-8, tol_abs=10**-8, output=""):
    """Méthode de recherche d'une racine de la fonction vectorielle f via la méthode de Newton avec approximation de la jacobienne :
        - x_0 donné,
        - x_kp1 = xk - Jac(f)(x_k)^-1*f(x_k).
    
    Les arguments attendus sont :
        - une fonction   f, admettant en entrée un vecteur x et renvoyant un  vecteur f(x),
        - une fonction jac, admettant en entrée un vecteur x et renvoyant une matrice Jac(f)(x),
        - un scalaire   x0 (de type int, float ou np.float64), point de départ de la méthode itérative.
    
    Les arguments optionnels sont :
        - un entier nb_iter (défaut = 100 ) définissant le nombre maximal d'itérations allouées à la méthode,
        - un réel   tol_rel (défaut = 1e-8) définissant la condition d'arrêt abs(x_k-x_km1) / (abs(x_k)+eps) <= tol_rel,
        - un réel   tol_abs (défaut = 1e-8) définissant la condition d'arrêt abs(f(x_k)) <= tol_abs,
        - une chaîne de caractères output (défaut = "") qui renvoie les affichages de la fonction vers :
            - la sortie standard si output = "pipe",
            - un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
            - nul part (aucune information écrite ni sauvegardée) si output = "" ou output = "None".

    La méthode vérifie les conditions suivantes :
         -   f est définie en x0, et renvoie un  vecteur de même dimension que x0,
         - jac est définie en x0, et renvoie une matrice carrée de dimension dim(x0)*dim(x0),
         - tous les paramètres reçus ont bien le type attendu.
    
    Les sorties de la méthode sont :
        - list_x, la liste des points x_k,
        - list_f, les valeurs par     f  des éléments de list_x,
        - list_d, les valeurs par Jac(f) des éléments de list_x.
        
    Exemples d'appel :
        - newton_nd(lambda x:x**2, lambda x: np.array([2*x]), np.array([1,1,1])),
        - def f(x):
              return(np.array([x[0]**2, x[1]/2, np.sin(x[2])]))
          def jac(x):
              return(np.array( [ [2*(x[0]), 0, 0], [0, 1/2, 0], [0, 0, np.cos(x[2])] ] ))
          x0 = np.array([1,1,1])
          newton_nd_avec_der(f, jac, x0).
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    check_parameters_consistency(f, jac, x0, nb_iter, tol_rel, tol_abs, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)
    
    # Initialisation de l'algorithme
    k, list_x, list_f, list_d = init_algo(f, jac, x0)
    write_iter(k, list_x, list_f)
    
    # Déroulement de l'algorithme
    while not(stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs)[0]):
        k, list_x, list_f, list_d = iter_algo(f, jac, k, list_x, list_f, list_d)
        write_iter(k, list_x, list_f)
    
    write_stopping(stopping_criteria(k, list_x, list_f, list_d, nb_iter, tol_rel, tol_abs)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, list_f, list_d)


