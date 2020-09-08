#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%##################################
# Import des bibliothèques requises #
#####################################

from MTH2210 import check_type_arguments, writing_function
import numpy as np



#%%########################################
# Fonction de vérification des paramètres #
###########################################

# Vérifie que le jeu de paramètres reçu par la méthode respecte les types attendus et les hypothèses mathématiques
def check_parameters_consistency(x, y, x_e, output):
    # Vérification des types des paramètres reçus
    params_array = [[x,       "x",       [list, np.ndarray]],
                    [y,       "y",       [list, np.ndarray]],
                    [x_e,     "x_e",     [list, np.ndarray]],
                    [output,  "output",   str]]
    if check_type_arguments.check_list(x)[0] == True:
        for i in range(len(x)):
            params_array.append([x[i], "xi", np.float64])
    if check_type_arguments.check_list(y)[0] == True:
        for i in range(len(y)):
            params_array.append([y[i], "yi", np.float64])
    if check_type_arguments.check_list(x_e)[0] == True:
        for i in range(len(x_e)):
            params_array.append([x_e[i], "x_evali", np.float64])
    params_array.append([output, "output", str])
    check_type_arguments.check_parameters(params_array)
    # Vérification de la cohérence des paramètres
    if len(x) != len(y):
        raise ValueError("Les dimensions de x (= "+str(len(x))+") et de y (= "+str(len(y))+") ne concordent pas")
    if len(x) != len(list(set(x))):
        raise ValueError("Le vecteur x des abscisses contient des doublons")



#%%########################################
# Fonctions de mise en page des résultats #
###########################################

# Crée la chaîne de caractères qui sera renvoyée pour chaque itération
def format_output(x, y, x_e, y_e):
    output_infos  = " {:^11} || {:^11}".format("x", "y")+"\n"
    output_infos += "-"*28 + "\n"
    for i in range(len(x)):
        output_infos += " {:>+8.4e} || {:>+8.4e}".format(x[i], y[i])+"\n"
    output_infos += "-"*28 + "\n"
    output_infos += " {:^11} || {:^11}".format("x_e", "y_e")+"\n"
    output_infos += "-"*28 + "\n"
    for i in range(len(x_e)):
        output_infos += " {:>+8.4e} || {:>+8.4e}".format(x_e[i], y_e[i])+"\n"
    return(output_infos)



#%%#######################################
# Fonctions d'itérations de l'algorithme #
##########################################

# Phase d'initialisation de toutes les suites exploitées par la méthode
def init_algo(x, y):
    def interpolation_1_output(x, y, xe):
        return(np.sum([ y[j]*np.prod([(x[i]-xe)/(x[i]-x[j]) for i in range(len(x)) if i != j]) for j in range(len(y)) ]) )
    def interpolation(x_e):
        if check_type_arguments.check_real(x_e)[0] == True:
            return(interpolation_1_output(x, y, np.array([x_e])))
        else:
            result = []
            for xe in x_e:
                result.append(interpolation_1_output(x, y, xe))
            return(np.array(result))
    return(interpolation)



#%%#####################################
# Définition de la fonction principale #
########################################

def lagrange(x, y, x_e, output=""):
    """Calcul du polynôme d'interpolation de Lagrange passant par tous les points (x_k,y_k) donnés en paramètres x et y :
        pour tout z, Lagrange(x,y)(z) = somme(j)(  y_j * prod(i)((x_i-z)/(x_i-x_j))  ).
    
    Les arguments attendus sont :
        un vecteur   x, contenant les abscisses des points d'interpolation,\n
        un vecteur   y, contenant les ordonnées des points d'interpolation,\n
        un vecteur x_e, contenant les abscisses des points auxquels le polynôme d'interpolation sera évalué.
    
    L'argument optionnel est :
        une chaîne de caractères output qui renvoie les affichages de la fonction vers :
            la sortie standard si output = "",\n
            un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),\n
            nul part (aucune information écrite ni sauvegardée) si output = "None".

    La méthode vérifie les conditions suivantes :
        x et y ont même dimension,\n
        x, y et x_e contiennent des réels,\n
        x ne contient pas deux fois la même abscisse,\n
        tous les paramètres reçus ont bien le type attendu.
    
    Les sorties de la méthode sont :
        y_e, la liste des valeurs du polynôme aux abscisses x_e,\n
        interpolation, une fonction renvoyant les valeurs du polynôme en chacun des éléments du vecteur d'abscisses qu'on lui passe en paramètre.
        
    Exemples d'appel :
        lagrange([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]),\n
        lagrange(np.array([-1,0,1]), np.array([0,-1,0]), np.array([-2,-1,-0.5,0,0.5,1,2])),\n
        y_e, lag = lagrange([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]), puis lag(np.array([3,4,5])).
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    check_parameters_consistency(x, y, x_e, output)
    write_output, _ = writing_function.define_writing_function(format_output, output)
    
    # Initialisation de l'algorithme
    interpolation = init_algo(x, y)
    
    # Déroulement de l'algorithme
    y_e = interpolation(x_e)
    write_output(x, y, x_e, y_e)
    
    # Renvoi de la liste des images des points x_e par le polynôme d'interpolation, et le polynôme comme une fonction
    return(y_e, interpolation)


