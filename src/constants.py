import os
from pathlib import Path

'''
Constants used anywhere in the program.
'''

treeNames_l = {"pAbies": "Tanne", "pAcer": "Ahorn",
               "pAll": "Gesamt", "pAlnus": "Erle", "pBetula": "Birke", "pCarpinus": "Hainbuche",
               "pCastanea": "Kastanie",
               "pConifer": "Nadelholz", "pDecid": "Laubholz", "pDouglas": "Douglasie", "pEurop": "Europa",
               "pFagus": "Buche",
               "pFraxinus": "Esche", "pHaupt": "Haupt", "pLarix": "Laerche", "pMalus": "Malus",
               "pNoEurop": "NichtEuropa",
               "pNoEuropD": "NichtEuropaDouglas", "pOtherLH": "OtherLH", "pOtherLN": "OtherLN", "pOtherNb": "OtherNb",
               "pPicea": "Fichte", "pPinus": "Kiefer", "pPopulus": "Pappel", "pPrunus": "Kirsche", "pPyrus": "Birne",
               "pQuerus": "Eiche", "pRobinia": "Robinie", "pSalix": "Weiden", "pSorbus": "Vogelbeere",
               "pTilia": "Linde",
               "pUlmus": "Ulme", "poHaupt": "poHaupt"}

hardwood = "Laubholz, Ahorn, Birne, Birke, Buche, Ebenholz, Eiche, Erle," \
           " Esche, Espe, Kirsche, Linde, Pappel, Robinie, Ulme, Walnuss," \
           " Weissbuche, Hainbuche, Kastanie, Weiden, Vogelbeere"

softwood = "Nadelholz, Fichte, Tanne, Laerche und Kiefer, Eibe, Douglasie, Pinie"

treeNames_g = {"Tanne": "pAbies", "Ahorn": "pAcer", "Gesamt": "pAll", "Erle": "pAlnus", "Birke": "pBetula",
               "Hainbuche": "pCarpinus", "Kastanie": "pCastanea", "Nadelholz": "pConifer", "Laubholz": "pDecid",
               "Douglasie": "pDouglas", "Europa": "pEurop", "Buche": "pFagus", "Esche": "pFraxinus", "Haupt": "pHaupt",
               "Laerche": "pLarix", "Malus": "pMalus", "NichtEuropa": "pNoEurop", "NichtEuropaDouglas": "pNoEuropD",
               "OtherLH": "pOtherLH", "OtherLN": "pOtherLN", "OtherNb": "pOtherNb", "Fichte": "pPicea",
               "Kiefer": "pPinus",
               "Pappel": "pPopulus", "Kirsche": "pPrunus", "Birne": "pPyrus", "Eiche": "pQuerus", "Robinie": "pRobinia",
               "Weiden": "pSalix", "Vogelbeere": "pSorbus", "Linde": "pTilia", "Ulme": "pUlmus", "poHaupt": "poHaupt"}

point_dist = 0.1
points_per_patch_sqrt = 10

cwd = os.getcwd()
pwd = str(Path(cwd).parent.absolute())
