# README
This is code originally from BIN lab 1: CGP [vasicek/courses/bin_lab1](https://www.fit.vutbr.cz/~vasicek/courses/bin_lab1/)

New commits contain my personal progress in making my own project to BIN [github.com/vlachvojta/BIN_CGP_with_GNN](https://github.com/vlachvojta/BIN_CGP_with_GNN).


## Lab progress - Velikost stavového prostoru

### Zadání
Vypočtěte velikost prohledávaného prostoru v případě, že používaté matici o rozměru 5x5 (5 sloupců, 5 řádků), každý element může být propojen s libovolným vstupem nebo výstupem předchozího sloupce (parametr l-back roven 1). Hledaný obvod má 5 vstupů a 1 výstup. Každý element rekonfigurovatelné matice může realizovat jednu ze 4 funkcí. Výstup obvodu může být připojen na výstup libovolného elementu.

### Rozpracování
- matice o rozměru 5x5 (5 sloupců, 5 řádků), 
- m = 5
- n = 5
- l-back = 1. 
- i = 5
- o = 1
- f = 4 (výběr 1 z 4 funkcí)

m*n     bloků
m*n*4   každý blok má 4 možnosti funkcí
m*n*4*20     každý blok má 5 možností na 2 vstupy (Variace s opakováním V_2(5) = 20)
m*n*4*20*m*n     nezávisle na sestavení ostatních věcí se výstup připojí na jeden z m*n výstupů bloků
m*n*4*20*m*n*2   2 možnosti výstupu
m*n*4*20*m*n*2*32  32 možností vstupů




