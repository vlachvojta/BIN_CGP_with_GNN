#define POPIS "#  doplni na násobek 2 3b cisel  6 vstupu, 6 vystupu\n#%%i i5, i4, i3, i2, i1, i0\n#%%o o5, o4, o3, o2, o1, o0\n"
//Pocet vstupu a vystupu
#define PARAM_IN 6        //pocet vstupu komb. obvodu
#define PARAM_OUT 6       //pocet vystupu komb. obvodu
//Inicializace dat. pole
#define init_data(a) \
  a[0]=0x00000000;\
  a[1]=0xffff0000;\
  a[2]=0xff00ff00;\
  a[3]=0xf0f0f0f0;\
  a[4]=0xcccccccc;\
  a[5]=0xaaaaaaaa;\
  a[6]=0x00000000;\
  a[7]=0xc0000000;\
  a[8]=0x38f00000;\
  a[9]=0xb4ccf000;\
  a[10]=0x66aacc00;\
  a[11]=0xaa00aa00;\
  a[12]=0xffffffff;\
  a[13]=0xffff0000;\
  a[14]=0xff00ff00;\
  a[15]=0xf0f0f0f0;\
  a[16]=0xcccccccc;\
  a[17]=0xaaaaaaaa;\
  a[18]=0xe0c08000;\
  a[19]=0x983870f0;\
  a[20]=0x54b46ccc;\
  a[21]=0x1e665aaa;\
  a[22]=0x66aacc00;\
  a[23]=0xaa00aa00;
//Pocet prvku pole
#define DATASIZE 24
