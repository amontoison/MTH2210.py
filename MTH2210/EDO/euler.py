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
def check_parameters_consistency(f, x0, t0, tm, m, output):
    # Vérification des types des paramètres reçus
    params_array = [[f,      "f",      types.FunctionType],
                    [x0,     "x0",     [np.ndarray, np.float64]],
                    [t0,     "t0",     np.float64],
                    [tm,     "tm",     np.float64],
                    [m,      "m",      np.int],
                    [output, "output", str]]
    check_type_arguments.check_parameters(params_array)
    # Vérification de la cohérence des paramètres
    try:
        f(x0, t0)
    except:
        raise ValueError("Fonction f non définie en (x0,t0)")
    try:
        f(x0, tm)
    except:
        raise ValueError("Fonction f non définie en (x0,tm)")
    if type(x0) == np.float64:
        if not(check_type_arguments.check_generic(f(x0,t0), np.float64)[0]):
            raise ValueError("f(x0,t0) n'est pas un vecteur np.float64 (type reçu :"+check_type_arguments.get_type(f(x0,t0))+")")
        if not(check_type_arguments.check_generic(f(x0,tm), np.float64)[0]):
            raise ValueError("f(x0,tm) n'est pas un vecteur np.float64 (type reçu :"+check_type_arguments.get_type(f(x0,tm))+")")
    else:
        if not(check_type_arguments.check_generic(f(x0,t0), np.ndarray)[0]):
            raise ValueError("f(x0,t0) n'est pas un vecteur np.ndarray (type reçu :"+check_type_arguments.get_type(f(x0,t0))+")")
        if not(check_type_arguments.check_generic(f(x0,tm), np.ndarray)[0]):
            raise ValueError("f(x0,tm) n'est pas un vecteur np.ndarray (type reçu :"+check_type_arguments.get_type(f(x0,tm))+")")
        if len(f(x0,t0)) != len(x0):
            raise ValueError("les dimensions de f(x0,t0) (= "+str(len(f(x0,t0)))+") et x0 (= "+str(len(x0))+") diffèrent")
    if m < 0:
        raise ValueError("Nombre d'itérations m défini à une valeur négative")



#%%########################################
# Fonctions de mise en page des résultats #
###########################################

# Crée la chaîne de caractères qui sera renvoyée pour chaque itération
def format_iter(k, x_k, t_k):

    if type(x_k) == np.float64:
        if k == 0:
            header  = "{:>4} || {:^11} | {:^8}"
            header  = header.format("k", "x_k", "t_k")
            header += "\n"
            header += "-"*(4+11+8 + 4+3)
            header += "\n"
        else:
            header = ""
        iter_infos = "{:>4} || {:>+11.4e} | {:>+9.4f}"
        iter_infos = iter_infos.format(k, x_k, t_k)
    
    else:
        if k == 0:
            n = len(x_k)
            len_str_xk = 2+11*n+2*(n-1)
            header  = "{:>4} || " + "{:^"+str(len_str_xk)+"}" + " | " + "{:^8}"
            header  = header.format("k", "x_k", "t_k")
            header += "\n"
            header += "-"*(4+len_str_xk+9 + 4+3)
            header += "\n"
        else:
            header = ""
        iter_infos  = "{:>4} || ".format(k)
        iter_infos += "["+", ".join(["{:>+11.4e}".format(xi) for xi in x_k])+"] | "+"{:>+9.4f}".format(t_k)
    
    return(header+iter_infos)



#%%########################################
# Fonctions de tests des critères d'arrêt #
###########################################

# Définit l'ensemble des critères d'arrêt possible, et les teste à chaque itération
def stopping_criteria(k, m):
    if k >= m:
        return(True, "Nombre maximal d'itérations k_max = {} atteint".format(m))
    return(False, "convergence inachevée")



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(x0, t0, tm, m):
    k = 0
    x = x0
    t = t0
    h = (tm-t0)/m
    list_x = [x]
    list_t = [t]
    return(k, x, t, h, list_x, list_t)

# Exécute une itération de la méthode
def iter_algo(f, k, x, t, h, list_x, list_t):
    k += 1
    x = x + h*f(x,t)
    t += h
    list_x.append(x)
    list_t.append(t)
    return(k, x, t, list_x, list_t)



#%%#####################################
# Définition de la fonction principale #
########################################

def euler(f, x0, t0, tm, m, output=""):
    """Méthode de résolution numérique d'une équation (dx/dt)(t) = f(x(t),t) par le schéma d'Euler :
        - x_0 donné, t_0 donné, pas de temps h donné,
        - x_kp1 = x_k + h*f(x_k,t_k),
        - t_kp1 = t_k + h.
    
    Les arguments attendus sont :
        - une fonction f, admettant en entrée un vecteur x et un réel t, renvoyant un vecteur f(x,t),
        - un vecteur  x0, condition initiale de l'équation,
        - deux réels  t0 et tm, les bornes de l'intervalle de temps sur lequel l'équation est appliquée,
        - un entier    m, le pas de discrétisation de [t0,tm], définissant donc h = (tm-t0)/m.
    
    L'argument optionnel est une chaîne de caractères output (défaut = "") qui renvoie les affichages de la fonction vers :
        - la sortie standard si output = "pipe",
        - un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),
        - nul part (aucune information écrite ni sauvegardée) si output = "" ou output = "None".

    La méthode vérifie les conditions suivantes :
        - la fonction f est définie en (x0,t0) et en (x0,tm),
        - f(x0,t0) renvoie un vecteur de la même dimension et même type que x0,
        - tous les paramètres reçus ont bien le type attendu.
    
    À noter que si x est un vecteur de dim 1, f doit être implémentée avec parcimonie pour ne pas renvoyer un mauvais type. Par exemple :
        - x = np.array(0) est un np.ndarray, x = np.array([0]) également,
        - f(x,t) = np.cos(np.array(0))   est un np.float64,
        - f(x,t) = np.cos(np.array([0])) est un np.ndarray,
        - f(x,t) = np.cos(t)             est un np.float64.
    Ces différences de types peuvent faire échouer la méthode si x est de dimension 1. La méthode est conçue pour fonctionner suivant :
        - si la dimension de x est > 1 :
            - x      défini par un np.array([coordonnées]),
            - f(x,t) renvoyant  un np.ndarray de même dimension que x,
        - si x est de dimension 1 :
            - x      défini par un np.float64(valeur), un float ou un int,
            - f(x,t) renvoyant  un np.float64,
        - cas sans garantie de fonctionnement correct :
            - x      complexe,
            - x      de dimension 1 défini par un np.array([valeur]).
    
    Les sorties de la méthode sont :
        - list_x, la liste des points x(t_k),
        - list_t, la liste des instants t_k.
        
    Exemples d'appel :
        - euler(lambda x,t : np.cos(t), np.float64(0), 0, 2*np.pi, 100),
        - euler(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 100),
        - def f(x,t):
              x0,x1 = 1,1
              return(np.array([x[0]*(x[1]-1),x[1]*(1-x[0])]))
          x = np.array([2,1])
          list_x, list_t = euler(f, x, 0, 10, 100).
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    if check_type_arguments.check_real(x0)[0]:
        x0 = np.float64(x0)
    check_parameters_consistency(f, x0, t0, tm, m, output)
    write_iter, write_stopping = writing_function.define_writing_function(format_iter, output)
    
    # Initialisation de l'algorithme
    k, x, t, h, list_x, list_t = init_algo(x0, t0, tm, m)
    write_iter(k, x, t)
    
    # Déroulement de l'algorithme
    while not(stopping_criteria(k, m)[0]):
        k, x, t, list_x, list_t = iter_algo(f, k, x, t, h, list_x, list_t)
        write_iter(k, x, t)
    
    write_stopping(stopping_criteria(k, m)[1])
    # Renvoi de la liste des approximations de la racine, des valeurs de f associées, et des erreurs relatives
    return(list_x, list_t)


