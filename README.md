# gated-tree-clip

おそらく実装済み
SyntacticDistanceGate
MultiheadAttentionWithGate
ResidualAttentionWithSyntacticDistanceBlock


未実装/未確認
GatedTreeTransformer
GatedTreeVisionTransformer
GatedTreeTextTransformer


# TODO:
- 自作したMultiheadAttentionの出力が、層を重ねるとnanになってしまう (nn.MultiheadAttentionでは起きない)
    - おそらく`in_proj_bias`の初期化忘れ、つまり`MultiheadAttention._reset_parameters()`中での
    ```python
    if self.in_proj_bias is not None:
        nn.init.constant_(self.in_proj_bias, 0.0)
    ```
    の忘れが原因 初期化は大切らしい
