
from exo1 import Resolution
from exo2 import DataTrans
import random

def script():

    print(""" 
    ###### AppDeCalcul  ######

    1. EXERCICE 1 : Calcul de R
    2. EXERCICE 2 : Calcul de la Fonction de DataTrans
    """
          )

    I = input("Entrer le numéro de l'exercice à calculer : ")

    if I == "1":
        try:
            n = input("Entrer la valeur de n : ")
            re = Resolution (int (n) )
            k = re.calcul()
            print("Le resulutat de la somme R est : ", k)
            
        except:
            print("Erreur lors du calcul de R(x) !")
            script()

        print("Voulez-vous reprendre le script ?")
        r = input("Oui ou Non : ")

        if r == "Oui":
            script()
        elif r == "Non":
            print("Merci d'avoir utilisé ce script !")
            exit()

    elif I == "2":
        n = input("Entrer le nombre de liste (n) : ")
        s = input("Entrer la taille des listes (s) : ")

        try:
            d = DataTrans(int (n), int (s))
            j = d.liste()
            print(j)
            print(d.listMin(j))
            print(d.listMax(j))
            mi = d.listMin(j)
            print(d.minGlobal(mi))
            ma = d.listMax(j)
            print (d.maxGlobal(ma))
            D = d.calcul_D(j)
            print(D)

        except :
            print("Il y a eu une erreur lors du calcul de la fonction f(D)")
            print("Veuillez réessayer !")
            script()

        # Demander à l'utilisateur de reprendre le script ou de le quitter

        print("Voulez-vous reprendre le script ?")
        r = input("Oui ou Non : ")

        if r == "Oui":
            script()

        elif r == "Non":
            print("Merci d'avoir utilisé ce script !")
            exit()

    else:
        print("Veuillez entrer un numéro d'exercice valide !")
        script()
script()
