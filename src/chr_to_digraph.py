import re
import pydantic

import networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt


class Node(pydantic.BaseModel):
    node_id: int
    inputs: list[int]
    function: int

    @classmethod
    def from_str(cls, node: str):
        node_id, *inputs, function = re.match(r"\[(\d+)\](\d+),(\d+),(\d+)", node).groups()
        return cls(node_id=int(node_id), inputs=list(map(int, inputs)), function=int(function))


class Chromosome(pydantic.BaseModel):
    n_inputs: int
    n_outputs: int
    rows: int
    cols: int
    block_inputs: int
    lback: int
    blocks_used: int

    nodes: dict[int, Node]
    outputs: list[int]

    input_id: int = 30
    function_id: int = 0
    output_id: int = 20

    def to_nx(self):
        G = nx.DiGraph()

        for i in range(self.n_inputs):
            G.add_node(f'in{i}', type=self.input_id, function_id=self.input_id)

        for node_id, node in self.nodes.items():
            G.add_node(f'{node_id}', type=self.function_id, function_id=node.function)
            for input_order, input_id in enumerate(node.inputs):
                G.add_edge(self.id_to_in(input_id), f'{node_id}')  #, label=input_order)

        for output_id, input_id in enumerate(self.outputs, start=len(self.nodes)):
            G.add_node(f'out{output_id}', type=self.output_id, function_id=self.output_id)
            G.add_edge(self.id_to_in(input_id), f'out{output_id}')  # , label=0)

        return G

    def id_to_in(self, id):
        if id < self.n_inputs:
            return f'in{id}'
        return f'{id}'

    @classmethod
    def from_str(cls, chromosome_str: str):
        match_obj = re.match(r"\{(.*)\}\((.*)\)\(([^\)]+)\)", chromosome_str)
        if not match_obj:
            raise ValueError(f'Invalid chromosome string: {chromosome_str}')
        preamble, data, outputs = match_obj.groups()
        # print(f'Preamble: {preamble}, Data: {data}, Outputs: {outputs}')
        n_inputs, n_outputs, rows, cols, block_inputs, lback, blocks_used = list(map(int, preamble.split(",")))
        # print(f'Preamble: {n_inputs=}, {n_outputs=}, {rows=}, {cols=}, {block_inputs=}, {lback=}, {blocks_used=}')

        nodes = {}
        for node in data.split(")("):
            # print(f'Parsing node: {node}')
            node = Node.from_str(node)
            # print(f'Parsed node: {node}')
            nodes[node.node_id] = node

        outputs = list(map(int, outputs.split(",")))
        # print(f'outputs: {outputs}')

        return cls(n_inputs=n_inputs, n_outputs=n_outputs, rows=rows, cols=cols, block_inputs=block_inputs, 
                   lback=lback, blocks_used=blocks_used, nodes=nodes, outputs=outputs)

def chr_to_digraph(chromosome_str: str):
    chromosome = Chromosome.from_str(chromosome_str).to_nx()
    data = torch_geometric.utils.from_networkx(chromosome)
    x = torch.cat((data.type.unsqueeze(1), data.function_id.unsqueeze(1)), dim=1).float()
    data = torch_geometric.data.Data(x=x, edge_index=data.edge_index)
    return data


def main():
    # test run
    data = chr_to_digraph('{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]9,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)')
    print(f'data: {data}')
    print(f'x: {data.x.shape}')
    print(f'edge_index: {data.edge_index.shape}')


if __name__ == '__main__':
    main()