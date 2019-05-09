
#define a generic class node that extends object class

'''
**  Each node receives input from multiple other nodes. These nodes are called inbound nodes
**  Each node creates a single output. This output can be passed to multiple other nodes. These nodes are called
    outbound nodes.
**  These inbound and outbound nodes are store in two different list of nodes.
**  A node can be treated as an element of the outbound node list of all of its inbound node
**  Inbound nodes list is empty for input layer and outbound node list is empty for output layer
**  Each node will calculate a value that represents its output
'''


class Node(object):

    def __init__(self, inbound_nodes = []):
        self.inbound_nodes = inbound_nodes          # assigning the empty inbound nodes to class level variable

        self.outbound_nodes = []                    # creating an empty outbound nodes

        for node in inbound_nodes:
            node.outbound_nodes.append(self)

        self.value = None                           # value is the calculated output of each node


    def forward_propagation(self):

        '''

        Forward propagation code goes here

        :return:
        '''

'''

**  Input nodes are special nodes that has no input that is, it does not require inbound nodes.
**  As Input nodes are Node, so we can extend Node class to implement input class
**  Input node is the only node where the value may be passed as an argument to forward() method
**  All other node implementation should get the value of the previous node from self.inbound_nodes
**  For example : val0 = self.inbound_nodes[0].value
**  Unlike the other sub classes of Node, the Input subclass does not actually calculate anything. It just holds a value
**  The value can be set explicitly or with the forward() method. This value is then fed through the rest of the network
'''


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward_propagation(self, value=None):
        if value is not None:
            self.value = value


'''
**  'Add' is another subclass of Node class that can perform a calculation 

'''


class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x,y])

    def forward_propagation(self):

        '''

        :return:
        '''

