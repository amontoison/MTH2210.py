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
def check_parameters_consistency(f, x0, nb_iter, tol_rel, tol_abs, output):
    # Vérification des types des paramètres reçus
    params_array = [[f,       "f",       types.FunctionType],
                    [x0,      "x0",      [np.ndarray, np.float64]],
                    [nb_iter, "nb_iter", int],
                    [tol_rel, "tol_rel", np.float64],
                    [tol_abs, "tol_abs", np.float64],
                    [output,  "output",  str]]
    check_type_arguments.check_parameters(params_array)
    # Vérification de la cohérence des paramètres
    try:
        f(x0)
    except:
        raise ValueError("Fonction f non définie en x0")
    if type(x0) == np.float64:
        if not(check_type_arguments.check_generic(f(x0), np.float64)[0]):
            raise ValueError("f(x0) n'est pas un vecteur np.float64 (type reçu :"+check_type_arguments.get_type(f(x0))+")")
    else:
        if not(check_type_arguments.check_generic(f(x0), np.ndarray)[0]):
            raise ValueError("f(x0) n'est pas un vecteur np.ndarray (type reçu :"+check_type_arguments.get_type(f(x0))+")")
        if len(f(x0)) != len(x0):
            raise ValueError("Les dimensions de f(x0) (= "+str(len(f(x0)))+") et x0 (= "+str(len(x0))+") diffèrent")
#    norme_iter_0 = np.linalg.norm(f(x0)-x0)
#    norme_iter_1 = np.linalg.norm(f(f(x0))-f(x0))
#    if norme_iter_0 < norme_iter_1:
#            raise ValueError("La fonction ne semble pas contractante : norm(f(x0)-x0) = "+str(norme_iter_0)+" < norm(f(f(x0)-f(x0))) = "+str(norme_iter_1))
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
def format_iter(k, x_k, f_k):

    if type(x_k) == np.float64:
        if k == 0:
            header  = "{:>4} || {:^11} | {:^11} | {:^11}"
            header  = header.format("k", "|f_k-x_k|", "x_k", "f_k")
            header += "\n"
            header += "-"*(4+11+11+11 + 4+3+3)
            header += "\n"
        else:
            header = ""
        iter_infos = "{:>4} || {:>+11.4e} | {:>+11.4e} | {:>+11.4e}"
        iter_infos = iter_infos.format(k, abs(x_k-f_k), x_k, f_k)

    else:
        if k == 0:
            n = len(x_k)
            len_str_xk = 2+11*n+2*(n-1)
            header  = "{:>4} || {:^13} | " + "{:^"+str(len_str_xk)+"}" + " | " + "{:^"+str(len_str_xk)+"}"
            header  = header.format("k", "norm(f_k-x_k)", "x_k", "f_k")
            header += "\n"
            header += "-"*(4+13+len_str_xk+len_str_xk + 4+3+3)
            header += "\n"
        else:
            header = ""
        iter_infos  = "{:>4} || ".format(k)
        iter_infos += "{:>+13.4e} | ".format(np.linalg.norm(x_k-f_k))
        iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in x_k])+"] | "
        iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in f_k])+"]"

    return(header+iter_infos)



#%%########################################
# Fonctions de tests des critères d'arrêt #
###########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(f, k, list_x, nb_iter, tol_rel, tol_abs):
    if k > nb_iter:
        return(True, "Nombre maximal d'itérations k_max={} autorisé dépassé".format(nb_iter))
    norm = np.linalg.norm(f(list_x[-1])-list_x[-1])
    if norm < tol_abs:
        return(True, "Point fixe localisé à {:7.1e} près : norme(f(x_k)-x_k) = {:10.4e}".format(tol_abs, norm))
    if k > 1:
        err_rel = check_relative_tolerance.tol_rel_approx(list_x[-1], list_x[-2])
        if err_rel < tol_rel:
            return(True, "Convergence de la méthode achevée à {:7.1e} près : erreur relative sur x = {:10.4e}".format(tol_rel, err_rel))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(f, x0):
    k = 0
    x = x0
    fx = f(x)
    list_x = [x]
    return(k, x, fx, list_x)

# Exécute une itération de la méthode
def iter_algo(f, k, x, list_x):
    k += 1
    x = f(x)
    fx = f(x)
    list_x.append(x)
    return(k, x, fx, list_x)



#%%#####################################
# Définition de la fonction principale #
########################################

def point_fixe(f, x0, nb_iter=100, tol_rel=10**-8, tol_abs=10**-8, output=""):
    """Méthode de calcul d'un point fixe x=f(x) par méthode itérative :
        - x_0 donné,
        - x_kp1 = f(x_k).

    Les arguments attendus sont :
        - une fonction f, admettant en entrée un vecteur x, renvoyant un vecteur f(x) de même dimension que x,
        - un vecteur x0, point de départ de la méthode itérative.

    Les arguments optionnels sont :
        - un entier nb_iter (défaut = 100 ) définissant le nombre maximal d'itérations allouées à la méthode,
        - un réel   tol_rel (défaut = 1e-8) définissant la condition d'arrêt abs(x_k-x_km1) / (abs(x_k)+eps) <= tol_rel,
        - un réel   tol_abs (défaut = 1e-8) définissant la condition d'arrêt abs(f(x_k)) <= tol_abs,
        - une chaîne de caractères output (défaut = "") qui renvoie les affichages de la fonction vers :
            - la sortie standard si output = "pipe",
            - un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
            - nul part (aucune information écrite ni sauvegardée) si output = "" ou output = "None".

    La méthode vérifie les conditions suivantes :
        - la fonction f est définie en x0,
        - norm(f(x0)-x0) >= norm(f(f(x0)-f(x0))) (/!\ vérification de cette condition enlevée en avril 2021 /!\)
        - f(x0) renvoie un vecteur de la même dimension que x0.

    À noter que si x est un vecteur de dim 1, f doit être implémentée avec parcimonie pour ne pas renvoyer un mauvais type. Par exemple, en définissant :
        - x = np.array(1) est un np.ndarray, x = np.array([1]) également,
        - f(x) = np.cos(x)           est un np.float64,
        - f(x) = np.array(np.cos(x)) est un np.ndarray,
        - f(x) = np.cos(t)           est un np.float64.
    Ces différences de types peuvent faire échouer la méthode si x est de dimension 1. La méthode est conçue pour fonctionner suivant :
        - si la dimension de x est > 1 :
            - x    défini par un np.array([coordonnées]),
            - f(x) renvoyant  un np.ndarray de même dimension que x,
        - si x est de dimension 1 :
            - x    défini par un np.float64(valeur), un float ou un int,
            - f(x) renvoyant  un np.float64,
        - cas sans garantie de fonctionnement correct :
            - x    complexe,
            - x    de dimension 1 défini par un np.array([valeur]).

    Les sorties de la méthode sont :
        - list_x,  la liste des points x_k,
        - err_rel, la liste des approximations de l'erreur relative à chaque itération (prenant pour référence le dernier élément de list_x).

    Exemples d'appel :
        - point_fixe(lambda x : x**2, 0.5),
        - point_fixe(lambda x : x**2, np.array([0.1,0.1])).
    """

    # Test des paramètres et définition de la destination de sortie des itérations
    if check_type_arguments.check_real(x0)[0]:
        x0 = np.float64(x0)
    check_parameters_consistency(f, x0, nb_iter, tol_rel, tol_abs, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)

    # Initialisation de l'algorithme
    k, x, fx, list_x = init_algo(f, x0)
    write_iter(k, x, fx)

    # Déroulement de l'algorithme
    while not(stopping_criteria(f, k, list_x, nb_iter, tol_rel, tol_abs)[0]):
        k, x, fx, list_x = iter_algo(f, k, x, list_x)
        write_iter(k, x, fx)
    err_rel = np.array( [abs(xi-list_x[-1]) for xi in list_x] )

    write_stopping(stopping_criteria(f, k, list_x, nb_iter, tol_rel, tol_abs)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, err_rel)
