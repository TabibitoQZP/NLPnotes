import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        # 使用restnet50作为特征提取器
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])

        # 使用2d卷积核, 注意用法, 其接受的输入是B, C, H, W, 可能很多人印象里是B, H, W, C
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads,
                                          num_encoder_layers, num_decoder_layers)
        # 可以看到, 分类比原本的多一个
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # 是的, 对于box来说, 就是拟合x1, y1, x2, y2四个数
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # 初始化3个可以调整的参数(原来是这样搞...), 后两个是位置编码
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # 这里相当于位置编码的前半部分来自row, 后半部分来自col
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8,
            num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200)
logits, bboxes = detr(inputs)
print(logits, bboxes)
