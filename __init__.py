#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%########################################
# Présentation de la bibliothèque MTH2210 #
###########################################

# Bibliothèque numérique du cours MTH2210 - Calcul scientifique pour ingénieurs de l'École Polytechnique de Montréal.
#
# Son contenu est le suivant :
# - fonctions de résolution d'équations différentielles (dx/dt)(t) = f(x(t),t) :
#       - méthode d'Euler,
#       - méthode de Runge-Kutta d'ordre 4,
# - fonctions d'interpolation d'un ensemble de points (x_i,y_i) :
#       - interpolation de Lagrange,
#       - interpolation par splines cubiques (avec diverses conditions aux bornes),
# - fonctions de recherche d'une racine d'une fonction scalaire :
#       - recherche dichotimique (bissection),
#       - méthode de Newton,
#       - méthode de la sécante,
# - fonction de calcul d'un point fixe d'une fonction scalaire :
#       - méthode du point fixe,
# - fonctions techniques invisibles pour l'utilisateur, exploitées par les fonctions principales :
#       - check_relative_tolerance(e1,e2) calcule le ratio abs(e1-e2) / (abs(e1)+epsilon_machine),
#       - check_type_arguments contient des fonctions vérifiant si le type des arguments reçus correspond au type attendu,
#       - writing_function définit les fonctions d'écriture des résultats (dans le stdout, dans un fichier ou nul part).
#
# Pour chacune des fonctions principales, une documentation est disponible dans le code.
# On peut également y accéder comme on accède à la documentation de n'importe quelle fonction Python.
#
# La bibliothèque a été codée sous Python 3.7.4.
# Elle exploite la bibliothèque Numpy; le code a été testé avec la version 1.18.1.
#
# Créée par Pierre-Yves Bouchet.



#%%##################################
# Import des bibliothèques requises #
#####################################

from MTH2210.bissection import bissection
from MTH2210.euler      import euler
from MTH2210.lagrange   import lagrange
from MTH2210.newton_1d  import newton_1d
from MTH2210.point_fixe import point_fixe
from MTH2210.rk4        import rk4
from MTH2210.secante    import secante
from MTH2210.spline3    import spline3



#%%#####################################
# Exemples d'utilisation des fonctions #
########################################

# Copier-coller ce code dans un fichier .py quelconque,
# et changer le chemin d'accès dans la ligne sys.path.insert
# pour la faire pointer vers le dossier contenant la bibliothèque.
# E.g., si la bibliothèque se trouve dans le dossier localisé en
#       /chemin/acces/vers/la/biblio/MTH2210,
# alors le chemin à ajouter dans la ligne sys.path.insert sera
#       /chemin/acces/vers/la/biblio.

# import sys
# sys.path.insert(0, '/chemin/acces/vers/la/biblio')
# import MTH2210
# import numpy as np

print("############################")
print("# Méthode de la bissection #")
print("############################")
print()
print("x, f = MTH2210.bissection(lambda x : np.sin(x), -0.5, 1/3) :\n")
x, f = MTH2210.bissection(lambda x : np.sin(x), -0.5, 1/3)
print("Les sorties sont :")
print("x =", x)
print("f =", f)
print()

# print("\n\n")

# print("############################")
# print("# Méthode d'approx d'Euler #")
# print("############################")
# print()
# print("cas x de dim 1 : x, t = MTH2210.euler(lambda x,t : np.cos(t), 0, 0, 2*np.pi, 10) :\n")
# x, t = MTH2210.euler(lambda x,t : np.cos(t), 0, 0, 2*np.pi, 10)
# print("Les sorties sont :")
# print("x =", x)
# print("t =", t)
# print("\n")
# print("cas x de dim 2 : x, t = MTH2210.euler(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 100 :\n")
# x, t = MTH2210.euler(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 10)
# print("Les sorties sont :")
# print("x =", x)
# print("t =", t)
# print()

# print("\n\n")

# print("############################")
# print("#  Interpolation Lagrange  #")
# print("############################")
# print()
# print("y, l = MTH2210.lagrange([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]) :\n")
# y, l = MTH2210.lagrange([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2])
# print("Les sorties sont :")
# print("y =", y)
# print("l =", l)
# print("La fonction l s'utilise comme suit :")
# print("l(5) =", l(5))
# print("l(np.array([5,10,100])) =", l(np.array([5,10,100])))
# print()

# print("\n\n")

# print("############################")
# print("#   Méthode de Newton 1D   #")
# print("############################")
# print()
# print("x, f, d = MTH2210.newton_1d(lambda x : np.sin(x), lambda x:np.cos(x), 1) :\n")
# x, f, d = MTH2210.newton_1d(lambda x : np.sin(x), lambda x:np.cos(x), 1)
# print("Les sorties sont :")
# print("x =", x)
# print("f =", f)
# print("d =", d)
# print()

# print("\n\n")

# print("############################")
# print("# Recherche du point fixe  #")
# print("############################")
# print()
# print("cas x de dim 1 : x = MTH2210.point_fixe(lambda x :x**2, 0.5) :\n")
# x = MTH2210.point_fixe(lambda x :x**2, 0.5)
# print("La sortie est :")
# print("x =", x)
# print("\n")
# print("cas x de dim 2 : x = MTH2210.point_fixe(lambda x : x**2, np.array([0.1,0.1])) :\n")
# x = MTH2210.point_fixe(lambda x : x**2, np.array([0.1,0.1]))
# print("Les sorties sont :")
# print("x =", x)
# print("t =", t)
# print()

# print("\n\n")

# print("############################")
# print("# Méthode d'approx via RK4 #")
# print("############################")
# print()
# print("cas x de dim 1 : x, t = MTH2210.rk4(lambda x,t : np.cos(t), np.float64(0), 0, 2*np.pi, 10) :\n")
# x, t = MTH2210.rk4(lambda x,t : np.cos(t), np.float64(0), 0, 2*np.pi, 10)
# print("Les sorties sont :")
# print("x =", x)
# print("t =", t)
# print("\n")
# print("cas x de dim 2 : x, t = MTH2210.rk4(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 10) :\n")
# x, t = MTH2210.rk4(lambda x,t : np.array([np.cos(t),np.sin(t)]), np.array([0,0]), 0, 2*np.pi, 10)
# print("Les sorties sont :")
# print("x =", x)
# print("t =", t)
# print()

# print("\n\n")

# print("############################")
# print("#Interpolation spline deg 3#")
# print("############################")
# print()
# print("Cas par défaut, spline naturelle : y, s = MTH2210.spline3([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2]) :\n")
# y, s = MTH2210.spline3([-1,0,1], [0,-1,0], [-2,-1,-0.5,0,0.5,1,2])
# print("Les sorties sont :")
# print("y =", y)
# print("s =", s)
# print("La fonction s s'utilise comme suit :")
# print("s(5) =", s(5))
# print("s(np.array([5,10,100])) =", s(np.array([5,10,100])))
# print("\n")
# print("Les autres cas s'obtiennent via les paramètres optionnels suivant le 3e :")
# print("MTH2210.spline3(x, y, x_e, cond_g, val_g, cond_d, val_d)")
# print("Cf la doc de la fonction pour leur utilisation.")

# print("\n\n")

# print("############################")
# print("#  Méthode de la sécante   #")
# print("############################")
# print()
# print("x, f, d = MTH2210.secante(lambda x :x**2, 2, 1) :\n")
# x, f, d = MTH2210.secante(lambda x :x**2, 2, 1)
# print("Les sorties sont :")
# print("x =", x)
# print("f =", f)
# print("d =", d)
# print()


