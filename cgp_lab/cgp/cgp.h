#define POPULACE_MAX 6       //maximalni pocet jedincu populace
#define MUTACE_MAX 3         //max pocet genu, ktery se muze zmutovat behem jedne mutace (o 1 mensi!)

#define PARAM_M 7            //pocet sloupcu
#define PARAM_N 7            //pocet radku
#define L_BACK 2             //1 (pouze predchozi sloupec)  .. param_m (maximalni mozny rozsah);

#define PARAM_GENERATIONS 2000000 //5000000   //max. pocet generaci evoluce
#define PARAM_RUNS 10            //max. pocet behu evoluce
#define FUNCTIONS 4              //max. pocet pouzitych funkci bloku (viz fitness() )
#define PERIODICLOGG 1  //  (PARAM_GENERATIONS/2) //po kolika krocich se ma vypsat populace
#define PERIODIC_LOG    //xPERIODIC_LOG           //zda se ma vypisovat populace

//-----------------------------------------------------------------------
// Preddefinovani vstupnich hodnot a spravnych vystupnich hodnot
// mozno pouzit vygenerovany h soubor z t2bconv
//-----------------------------------------------------------------------
#include "configs/multi3.h"
