#define POPIS "#  doplni na sudou paritu4b  4 vstupy, 1 vystup\n#%%i i3,i2,i1,i0\n#%%o out\n"
//Pocet vstupu a vystupu
#define PARAM_IN 4        //pocet vstupu komb. obvodu
#define PARAM_OUT 1       //pocet vystupu komb. obvodu
//Inicializace dat. pole
#define init_data(a) \
  a[0]=0xffffff00;\
  a[1]=0xfffff0f0;\
  a[2]=0xffffcccc;\
  a[3]=0xffffaaaa;\
  a[4]=0x00006996;
//Pocet prvku pole
#define DATASIZE 5
