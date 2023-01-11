from torch import nn


class BasicLayer(nn.Module):
    def __init__(self, depth, is_sw):
        super(BasicLayer, self).__init__()
        self.blocks = nn.ModuleList()
        self.depth = depth
        self.is_sw = is_sw

    def append(self, block):
        self.blocks.append(block)

    def forward(self, x):
        if self.is_sw:
            for i in range(self.depth):
                x = self.blocks[0](x)
        else:
            for block in self.blocks:
                x = block(x)
        return x
    
    def get_depth(self):
        return len(self.blocks)