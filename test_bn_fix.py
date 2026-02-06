import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poseCtrl.models.pose_adaptor import PointNetEncoder, STNkd

def test_bn_stability():
    print("=" * 80)
    print("测试 BN 层数值稳定性")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 测试 STNkd
    print("\n" + "-" * 80)
    print("测试 STNkd")
    print("-" * 80)
    
    stn = STNkd(k=64).to(device)
    stn.train()
    
    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, 64, 13860).to(device)
    
    print(f"输入形状: {x.shape}")
    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"输入是否有NaN: {torch.isnan(x).any()}")
    
    # 前向传播
    try:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = stn(x)
        
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"输出是否有NaN: {torch.isnan(output).any()}")
        print(f"输出是否有Inf: {torch.isinf(output).any()}")
        
        # 检查 BN 统计量
        print("\nBN 统计量:")
        for i, bn in enumerate([stn.bn1, stn.bn2, stn.bn3, stn.bn4, stn.bn5], 1):
            print(f"  bn{i}: running_mean 是否有NaN: {torch.isnan(bn.running_mean).any()}")
            print(f"  bn{i}: running_var 是否有NaN: {torch.isnan(bn.running_var).any()}")
            if not torch.isnan(bn.running_mean).any():
                print(f"  bn{i}: running_mean 范围: [{bn.running_mean.min():.4f}, {bn.running_mean.max():.4f}]")
            if not torch.isnan(bn.running_var).any():
                print(f"  bn{i}: running_var 范围: [{bn.running_var.min():.4f}, {bn.running_var.max():.4f}]")
        
        print("✓ STNkd 测试通过")
    except Exception as e:
        print(f"✗ STNkd 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 PointNetEncoder
    print("\n" + "-" * 80)
    print("测试 PointNetEncoder")
    print("-" * 80)
    
    encoder = PointNetEncoder(channel=3).to(device)
    encoder.train()
    
    # 创建测试输入
    batch_size = 4
    points = torch.randn(batch_size, 3, 13860).to(device)
    V_matrix = torch.randn(batch_size, 4, 4).to(device)
    P_matrix = torch.randn(batch_size, 4, 4).to(device)
    text_feature = torch.randn(batch_size, 77, 768).to(device)
    
    print(f"points 形状: {points.shape}")
    print(f"V_matrix 形状: {V_matrix.shape}")
    print(f"P_matrix 形状: {P_matrix.shape}")
    print(f"text_feature 形状: {text_feature.shape}")
    
    # 前向传播
    try:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output, trans_feat, importance = encoder(points, V_matrix, P_matrix, text_feature)
        
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"输出是否有NaN: {torch.isnan(output).any()}")
        print(f"输出是否有Inf: {torch.isinf(output).any()}")
        
        print(f"\ntrans_feat 形状: {trans_feat.shape}")
        print(f"trans_feat 范围: [{trans_feat.min():.4f}, {trans_feat.max():.4f}]")
        print(f"trans_feat 是否有NaN: {torch.isnan(trans_feat).any()}")
        
        print(f"\nimportance 形状: {importance.shape}")
        print(f"importance 范围: [{importance.min():.4f}, {importance.max():.4f}]")
        print(f"importance 是否有NaN: {torch.isnan(importance).any()}")
        
        # 检查 BN 统计量
        print("\nPointNetEncoder BN 统计量:")
        for i, bn in enumerate([encoder.bn1, encoder.bn2, encoder.bn3], 1):
            print(f"  bn{i}: running_mean 是否有NaN: {torch.isnan(bn.running_mean).any()}")
            print(f"  bn{i}: running_var 是否有NaN: {torch.isnan(bn.running_var).any()}")
            if not torch.isnan(bn.running_mean).any():
                print(f"  bn{i}: running_mean 范围: [{bn.running_mean.min():.4f}, {bn.running_mean.max():.4f}]")
            if not torch.isnan(bn.running_var).any():
                print(f"  bn{i}: running_var 范围: [{bn.running_var.min():.4f}, {bn.running_var.max():.4f}]")
        
        print("\nSTNkd 内部 BN 统计量:")
        for i, bn in enumerate([encoder.fstn.bn1, encoder.fstn.bn2, encoder.fstn.bn3, 
                                encoder.fstn.bn4, encoder.fstn.bn5], 1):
            print(f"  fstn.bn{i}: running_mean 是否有NaN: {torch.isnan(bn.running_mean).any()}")
            print(f"  fstn.bn{i}: running_var 是否有NaN: {torch.isnan(bn.running_var).any()}")
            if not torch.isnan(bn.running_mean).any():
                print(f"  fstn.bn{i}: running_mean 范围: [{bn.running_mean.min():.4f}, {bn.running_mean.max():.4f}]")
            if not torch.isnan(bn.running_var).any():
                print(f"  fstn.bn{i}: running_var 范围: [{bn.running_var.min():.4f}, {bn.running_var.max():.4f}]")
        
        print("✓ PointNetEncoder 测试通过")
    except Exception as e:
        print(f"✗ PointNetEncoder 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_bn_stability()