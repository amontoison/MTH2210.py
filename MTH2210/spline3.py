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
def check_parameters_consistency(x, y, x_e, cond_g, val_g, cond_d, val_d, output):
    # Vérification des types des paramètres reçus
    params_array = [[x,       "x",       [list, np.ndarray]],
                    [y,       "y",       [list, np.ndarray]],
                    [x_e,     "x_e",     [list, np.ndarray]],
                    [cond_g,  "cond_g",   int],
                    [val_g,   "val_g",    int],
                    [cond_d,  "cond_d",   int],
                    [val_d,   "val_d",    int],
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
    x_sort = [xi for xi in x]
    x_sort.sort()
    if x != x_sort:
        raise ValueError("Le vecteur x des abscisses n'est pas dans l'ordre croissant")
    if cond_g not in [0,1,2,3]:
        raise ValueError("La condition limite à gauche n'a pas une valeur recevable (= "+str(cond_g)+"), attendue dans [0,1,2,3]")
    if cond_d not in [0,1,2,3]:
        raise ValueError("La condition limite à droite n'a pas une valeur recevable (= "+str(cond_d)+"), attendue dans [0,1,2,3]")



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

# Phase de calcul de la spline
def init_algo(x, y, c_g, v_g, c_d, v_d):
    
    def hx(i):
        return(x[i]-x[i-1])
    def hy(i):
        return(y[i]-y[i-1])
    
    np1 = len(x)
    n = np1-1
    
    A = np.zeros((np1,np1))
    b = np.zeros((np1,1))
    for i in range(1,n):
        A[i,i-1] = hx(i)   / (hx(i)+hx(i+1))
        A[i,i]   = 2
        A[i,i+1] = hx(i+1) / (hx(i)+hx(i+1))
        b[i]     = 6  *  ( hy(i+1)/hx(i+1) - hy(i)/hx(i) )  /  (hx(i)+hx(i+1))
    
    if c_g == 0:
        A[0,0] = 1
        b[0]   = 0
    elif c_g == 1:
        A[0,0] = 1
        b[0]   = v_g
    elif c_g == 2:
        A[0,0] = 1
        A[0,1] = -1
        b[0]   = 0
    elif c_g == 3:
        A[0,0] = 2
        A[0,1] = 1
        b[0]   = 6/hx(1) * (hy(1)/hx(1) - v_g)
    
    if c_d == 0:
        A[-1,-1] = 1
        b[-1]    = 0
    elif c_d == 1:
        A[-1,-1] = 1
        b[-1]    = v_d
    elif c_d == 2:
        A[-1,-1] = 1
        A[-1,-2] = -1
        b[-1]    = 0
    elif c_d == 3:
        A[-1,-1] = 2
        A[-1,-2] = 1
        b[-1]    = 6/hx(n) * (v_d - hy(n)/hx(n))
    
    d2f = np.linalg.solve(A,b)
    
    def interpolation_1_output(xe):
        i = 1
        while i < len(x) and xe > x[i]:
            i += 1
        if i == len(x):
            i -= 1
        coeff_1 = -1 * d2f[i-1] / (6*hx(i))
        coeff_2 = +1 * d2f[i  ] / (6*hx(i))
        coeff_3 = d2f[i-1]*hx(i)/6 - y[i-1]/hx(i)
        coeff_4 = y[i]/hx(i) - d2f[i]*hx(i)/6
        return(np.float64(coeff_1*(xe-x[i])**3 + coeff_2*(xe-x[i-1])**3  + coeff_3*(xe-x[i]) + coeff_4*(xe-x[i-1])))
    
    def interpolation(x_e):
        if check_type_arguments.check_real(x_e)[0] == True:
            return(interpolation_1_output(np.array([x_e])))
        else:
            result = []
            for xe in x_e:
                result.append(interpolation_1_output(xe))
            return(np.array(result))
    
    return(interpolation)



#%%#####################################
# Définition de la fonction principale #
########################################

def spline3(x, y, x_e, cond_g=0, val_g=0, cond_d=0, val_d=0, output=""):
    """Calcul d'une spline cubique d'interpolation de tous les points (x_k,y_k) donnés en paramètres x et y.
    
    Les arguments attendus sont :
        un vecteur x, contenant les abscisses des points d'interpolation,\n
        un vecteur y, contenant les ordonnées des points d'interpolation,\n
        un vecteur x_e, contenant les abscisses des points auxquels le polynôme d'interpolation sera évalué.
    
    Les arguments optionnels sont :
        un entier cond_g dans [0,1,2,3], déterminant la condition à gauche (défaut = 0) :
            cond_g = 0 impose la condition naturelle à gauche,\n
            cond_g = 1 impose à la courbure à gauche de valoir val_g,\n
            cond_g = 2 impose à la courbure d'être constante à gauche\n.
            cond_g = 3 impose à la pente à gauche de valoir val_g.
        un réel val_g, utilisé dans la détermination de la condition à gauche (ignorée si cond_g = 0 ou 2) (défaut = 0),\n
        un entier cond_g dans [0,1,2,3], déterminant la condition à droite (analogue à cond_g),\n
        un réel val_d, utilisé dans la détermination de la condition à droite (analogue à val_g),\n
        une chaîne de caractères output qui renvoie les affichages de la fonction vers :
            la sortie standard si output = "",\n
            un fichier ayant pour nom+extension output (le paramètre doit donc contenir l'extension voulue, et le chemin d'accès doit exister),\n
            nul part (aucune information écrite ni sauvegardée) si output = "None".

    La méthode vérifie les conditions suivantes :
        x et y ont même dimension,\n
        x, y et x_e contiennent des réels,\n
        x ne contient pas deux fois la même abscisse,\n
        les valeurs de x sont dans l'ordre croissant,\n
        cond_g et cond_d sont tous les deux dans [0,1,2,3],\n
        tous les paramètres reçus ont bien le type attendu.
    
    Les sorties de la méthode sont :
        y_e, la liste des valeurs du polynôme aux abscisses x_e,\n
        interpolation, une fonction renvoyant les valeurs de la spline en chacun des éléments du vecteur d'abscisses qu'on lui passe en paramètre.
        
    Exemples d'appel :
        spline3([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]),\n
        spline3(np.array([-1,0,1]), np.array([0,-1,0]), np.array([-2,-1,-0.5,0,0.5,1,2])),\n
        y_e, spl = spline3([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]), puis spl(np.array([3,4,5]))
    """
    
    # Test des paramètres et définition de la destination de sortie des itérations
    check_parameters_consistency(x, y, x_e, cond_g, val_g, cond_d, val_d, output)
    write_output, _ = writing_function.define_writing_function(format_output, output)
    
    # Initialisation de l'algorithme
    interpolation = init_algo(x, y, cond_g, val_g, cond_d, val_d)
    
    # Déroulement de l'algorithme
    y_e = interpolation(x_e)
    write_output(x, y, x_e, y_e)
    
    # Renvoi de la liste des images des points x_e par le polynôme d'interpolation, et le polynôme comme une fonction
    return(y_e, interpolation)


