#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 12:00:00 2020

@author: Pierre-Yves Bouchet
"""



#%%###################################################
# Définition de la fonction appelée par les méthodes #
######################################################

# Définit la fonction write_iter, qui écrira dans le stdout ou un fichier selon le contenu de output
def define_writing_function(format_iter, output):
    if output.lower() == "pipe":
        def write_iter(*args):
            print(format_iter(*args))
        def write_stopping(reason):
            print(reason+"\n")
    elif output.lower() in ["none", ""]:
        def write_iter(*args):
            pass
        def write_stopping(reason):
            pass
    else:
        def write_iter(*args):
            with open(output, "a+") as file:
                file.write(format_iter(*args)+"\n")
        def write_stopping(reason):
            with open(output, "a+") as file:
                file.write(reason+"\n")
    return(write_iter, write_stopping)


