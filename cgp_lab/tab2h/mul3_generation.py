
# var = 2
# print(var)
# print(f'{var:b}')

n_bits = 5
n_inputs = n_bits * 2
n_outputs = n_bits * 2

print(f'# doplni na násobek 2 {n_bits}b cisel')
print(f'# {n_inputs} vstupu, {n_outputs} vystupu')
inputs = ', '.join([f'i{i}' for i in range(n_inputs-1, -1, -1)])
print(f"#%i {inputs}")
outputs = ', '.join([f'o{i}' for i in range(n_outputs-1, -1, -1)])
print(f"#%o {outputs}")

for var1 in range(2**n_bits):
    for var2 in range(2**n_bits):
        krat = var1 * var2
        print(f'{var1:05b}{var2:05b} : {krat:010b}')


