#define POPIS "#  doplni na soucet 2 2b cisel  4 vstupy, 3 vystupy\n#%%i i3,i2,i1,i0\n#%%o o2, o1, o0\n"
//Pocet vstupu a vystupu
#define PARAM_IN 4        //pocet vstupu komb. obvodu
#define PARAM_OUT 3       //pocet vystupu komb. obvodu
//Inicializace dat. pole
#define init_data(a) \
  a[0]=0xffffff00;\
  a[1]=0xfffff0f0;\
  a[2]=0xffffcccc;\
  a[3]=0xffffaaaa;\
  a[4]=0xffffec80;\
  a[5]=0xffff936c;\
  a[6]=0x00005a5a;
//Pocet prvku pole
#define DATASIZE 7
