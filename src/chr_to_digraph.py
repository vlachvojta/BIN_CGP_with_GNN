import re
import pydantic

import networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum

class NodeType(Enum):
    INPUT = 0
    FUNCTION = 1
    OUTPUT = 2

class Node(pydantic.BaseModel):
    node_id: int
    inputs: list[int]
    function: int

    def do_node_function(self, in1, in2):
        match self.function:
            case 0:
                return in1
            case 1:
                return in1 & in2
            case 2:
                return in1 | in2
            case 3:
                return in1 ^ in2

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
    # sorted_node_ids: list[int] = []  # TODO save here
    outputs: list[int]

    input_id: int = -10
    function_id: int = 1
    output_id: int = 20

    def to_nx(self, features: int = 2):
        G = nx.DiGraph()

        for i in range(self.n_inputs):
            G.add_node(f'in{i}', x=[NodeType.INPUT.value, self.input_id, self.node_depth(i), self.node_col(i)], node_type=NodeType.INPUT.value) #type=self.input_id, function_id=self.input_id)

        for node_id, node in self.nodes.items():
            G.add_node(f'{node_id}', x=[NodeType.FUNCTION.value, node.function, self.node_depth(node_id), self.node_col(node_id)], node_type=NodeType.FUNCTION.value) #type=self.function_id, function_id=node.function)  # TODO add depth feature
            for input_order, input_id in enumerate(node.inputs):
                G.add_edge(self.id_to_in(input_id), f'{node_id}')  #, label=input_order)

        for output_id, input_id in enumerate(self.outputs, start=len(self.nodes)):
            G.add_node(f'out{output_id}', x=[NodeType.OUTPUT.value, self.output_id, self.node_depth(input_id) + 1, self.cols + 1], node_type=NodeType.OUTPUT.value)  #) type=self.output_id, function_id=self.output_id)
            G.add_edge(self.id_to_in(input_id), f'out{output_id}')  # , label=0)

        return G

    def node_depth(self, node_id):
        if node_id < self.n_inputs:
            return 0

        return 1 + max([self.node_depth(input_id) for input_id in self.nodes[node_id].inputs])

    def node_col(self, node_id):
        if node_id < self.n_inputs:
                return 0

        return ((node_id - self.n_inputs) // self.rows) + 1

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
        n_inputs, n_outputs, rows, cols, block_inputs, lback, blocks_used = list(map(int, preamble.split(",")))

        nodes = {}
        for node in data.split(")("):
            node = Node.from_str(node)
            nodes[node.node_id] = node

        outputs = list(map(int, outputs.split(",")))

        return cls(n_inputs=n_inputs, n_outputs=n_outputs, rows=rows, cols=cols, block_inputs=block_inputs, 
                   lback=lback, blocks_used=blocks_used, nodes=nodes, outputs=outputs)

    def simulate_input(self, inputs):
        if not len(inputs) == self.n_inputs:
            raise ValueError(f'Unexpected length of inputs. Expected: {self.n_inputs}, but got: {len(inputs)}')
        # def get_in_or_node(self, inputs, nodes):

        values = {input_id: value  for input_id, value in enumerate(inputs)}  # index is node_id (including inputs)

        # pass through function nodes
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            values[node_id] = node.do_node_function(values[node.inputs[0]], values[node.inputs[1]])
            # print(f'[{node_id}] {values[node.inputs[0]]} (id: {node.inputs[0]}) FCE {values[node.inputs[1]]} (id: {node.inputs[1]}) = {values[node_id]}')

        return [values[node_id] for node_id in self.outputs]


# class ChromosomeSimulator:
#     def __init__(self, chromosome: Chromosome, parallel_input: bool = False, usefull_nodes_only: bool = False):


#     def __call__(self, sim_input):
#         if isinstance(sim_input)
#         # if parallel input
#         #   self.simulate_input_parallel(input)
#         # else



def chr_to_digraph(chromosome_str: str):
    chromosome = Chromosome.from_str(chromosome_str).to_nx()
    data = torch_geometric.utils.from_networkx(chromosome)
    return data


# def testing():
#     chromosome_str = '{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]9,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)'
#     chromosome = Chromosome.from_str(chromosome_str).to_nx()
#     data = torch_geometric.utils.from_networkx(chromosome)
#     print(f'data: {data}')
#     print(f'node_type: {data.node_type}')
#     # print(f'x: {data.x}')
    

def test_ch_to_digraph():
    # test run
    data = chr_to_digraph('{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]9,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)')
    print(f'data: {data}')
    print(f'x: {data.x.shape}')
    print(f'edge_index: {data.edge_index.shape}')

def test_input_simulation():
    chromosome = Chromosome.from_str('{6,6, 7,7, 2,2,29}([6]1,5,1)([7]2,4,1)([8]4,0,1)([9]5,2,1)([10]4,1,1)([11]1,3,2)([12]0,0,0)([13]11,7,1)([14]3,2,1)([15]0,5,1)([16]9,10,3)([17]6,7,1)([18]3,1,1)([19]8,2,2)([20]8,18,2)([21]14,5,3)([22]15,10,3)([23]3,0,1)([24]17,14,3)([25]7,6,3)([26]10,2,0)([27]23,16,1)([28]24,22,3)([29]0,19,0)([30]15,14,2)([31]5,4,2)([32]0,24,2)([33]20,17,2)([34]1,27,0)([35]33,27,3)([36]27,33,1)([37]30,26,1)([38]23,23,0)([39]20,25,3)([40]22,0,1)([41]37,38,2)([42]34,31,3)([43]35,37,3)([44]35,2,1)([45]2,5,3)([46]36,37,2)([47]40,35,1)([48]39,47,1)([49]36,41,3)([50]41,34,0)([51]0,41,1)([52]46,38,1)([53]47,47,1)([54]1,37,2)(52,49,43,28,25,9)')
    outputs = chromosome.simulate_input(np.array([0, 1, 0, 0, 1, 0])) #  2(010)*2(010) = 4(000100)
    print(outputs)

def test_net_to_png():
    chromosome_str = '{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]9,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)'
    chromosome_nx = Chromosome.from_str(chromosome_str).to_nx()
    nx.draw(chromosome_nx, with_labels=True)
    plt.savefig('net.png')
    # plt.show()



if __name__ == '__main__':
    test_ch_to_digraph()
    # test_input_simulation()
    # test_net_to_png()
