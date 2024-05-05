# NOTES

## Commands
generating cgp chromosomes: `cgp_lab/cgp: make && ./cgp | head -10000 > evolution.log ; head -100 evolution.log > evolution_short.log`

## Default function blocks
vstupy: in1, in2
vystup: out

- 0 - wire in1 -> out
- 1 - AND
- 2 - OR
- 3 - XOR
- 4
- 5
- 6
- 7
- 8
- 9

## BIT squeezing
```
00000000000001010110110001100011  LSB
00000000000100101101011010000111   |
10100010111111100010111110011100   V
01001001100101100000101000010111   A
10010111000100100101100010110101   |
01001101011010011001110110111110  MSB
MSB ->                   <- LSB
```

## Vypocet Fitness 
### v CGP z labiny
param_fitev = DATASIZE / (param_in + param_out); //Spocitani poctu pruchodu pro ohodnoceni
              24       / (6        + 6) = 2
maxfitness = param_fitev*param_out*32;         //Vypocet max. fitness
             2          *6        *32=384

### Moje
2 ^ 6 vstupu může mít 2 ^ 6 výstupů


## Výsledky Bayes inference
s Bayes: 17738 simulací
normální: 19968 simulací