
# var = 2
# print(var)
# print(f'{var:b}')


for var1 in range(8):
    for var2 in range(8):
        result = var1 + var2
        print(f'{var1:03b}{var2:03b} : {result:06b}')


