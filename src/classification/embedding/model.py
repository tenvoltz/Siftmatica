import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEmbeddingCNN(nn.Module):
    """
    Lightweight CNN for learning 64-dimensional texture embeddings.
    
    Architecture:
    - 3 convolution blocks (Conv2d + ReLU + BatchNorm)
    - 1 max-pooling layer (after first conv block)
    - Global average pooling
    - Final linear projection to 64D embedding space
    
    Input: (batch, 3, 16, 16) RGB texture normalized to [0, 1]
    Output: (batch, 64) L2-normalized embedding vector
    
    Args:
        embedding_dim (int): Dimension of output embedding. Default: 64
        in_channels (int): Number of input channels. Default: 3
        hidden_dims (list): Hidden dimensions for conv blocks. Default: [32, 64, 128]
    """
    
    def __init__(self, embedding_dim: int = 64, in_channels: int = 3, 
                 hidden_dims: list = None):
        super(TinyEmbeddingCNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        self.embedding_dim = embedding_dim
        
        # Conv Block 1 + Max Pool
        self.conv1 = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        self.relu2 = nn.ReLU(inplace=True)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])
        self.relu3 = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final linear projection
        self.embedding_projection = nn.Linear(hidden_dims[2], embedding_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding_projection(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


def create_embedding_model(embedding_dim: int = 64, in_channels: int = 3) -> TinyEmbeddingCNN:
    return TinyEmbeddingCNN(embedding_dim=embedding_dim, in_channels=in_channels)


if __name__ == "__main__":
    model = create_embedding_model()
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 16, 16)
    embeddings = model(x)
    
    norms = torch.norm(embeddings, p=2, dim=1)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embedding norms (should be 1): {norms}")
    print(f"Total model parameters: {num_params:,}")
    
