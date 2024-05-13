# BIN_CGP_with_GNN
BIN project (Bio-Inspired Computers) at FIT (B|V)UT. 2023/2024 summer semestr.


V tomto projektu jsem se snažil odhadovat fitnes a počet použítých bloků pomocí Grafové neuronové sítě. Nepodařilo se mi dosáhnout uspokojivých výsledků, proto jsem zkusil ještě variantu eliminace simulací za pomocí Bayesovského modelu Beta rozložení modelující pravděpodobný výskyt výsledné fitness (zde jako 0 až 1, značí poměr správných výstupů vůči celku)

Experimenty jsem prováděl pro jednoduchost pouze pro 3 bitovou násobičku, kde grafové sítě nezafungovali pro různé velikosti datasetu a nezvládly se naučit užitečné vlastnosti struktury obvodu.

Pro Bayesovský model jsem provedl experimenty ověřující to, že pro takto malý obvod lze pomocí odhadu pravděpodobnosti výskytu skutečných fitness ušetřit 12 % simulací (za cenu počítání integrálů). Pro větší obvody s více hodnotami vstupů se bude nejspíš počet ušetřených simulací zvedat.
