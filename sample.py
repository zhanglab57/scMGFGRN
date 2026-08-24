import pandas as pd
import numpy as np
import random
from itertools import product

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)

def create_negative_samples(positive_df, gene_list=None, allow_self_loop=False):
    """
    为基因调控网络构建负样本
    
    参数:
    positive_df: pandas DataFrame, 正样本数据，至少包含两列表示基因对
    gene_list: list, 所有基因的列表（可选）。如果为None，则从正样本中提取所有基因
    allow_self_loop: bool, 是否允许自环（基因对自身）。默认为False
    
    返回:
    negative_df: pandas DataFrame, 负样本数据
    """
    
    # 确保正样本数据有两列
    if len(positive_df.columns) < 2:
        raise ValueError("正样本DataFrame至少需要两列来表示基因对")
    
    # 获取基因对列名
    gene_col1, gene_col2 = positive_df.columns[:2]
    
    # 如果没有提供基因列表，从正样本中提取所有唯一基因
    if gene_list is None:
        tf_list = list(set(positive_df[gene_col1].tolist()))
        gene_list = list(set((positive_df[gene_col2].tolist())))

    # 创建所有可能的基因对（包括自环或不包括，根据参数）
    if allow_self_loop:
        all_possible_pairs = list(product(gene_list, repeat=2))
    else:
        all_possible_pairs = [(g1, g2) for g1 in tf_list for g2 in gene_list if g1 != g2]
    
    # 将正样本转换为集合以便快速查找
    positive_pairs = set(positive_df.apply(lambda row: tuple(row[:2]), axis=1))
    
    # 获取所有非正样本的基因对
    negative_candidates = [pair for pair in all_possible_pairs if pair not in positive_pairs]
    
    # 检查是否有足够的负样本候选
    n_positive = len(positive_df)
    if len(negative_candidates) < n_positive:
        print(f"警告: 负样本候选数量({len(negative_candidates)})少于正样本数量({n_positive})")
        print("将使用所有可用的负样本候选")
        n_samples = min(n_positive, len(negative_candidates))
    else:
        n_samples = n_positive
    
    # 随机抽取负样本
    selected_negative_pairs = random.sample(negative_candidates, n_samples)
    
    # 创建负样本DataFrame
    negative_df = pd.DataFrame(selected_negative_pairs, columns=[gene_col1, gene_col2])
    
    # 如果正样本有其他列，可以在负样本中添加对应的占位符列
    if len(positive_df.columns) > 2:
        for col in positive_df.columns[2:]:
            # 对于数值列，可以填充0；对于分类列，可以填充空值或特定值
            if pd.api.types.is_numeric_dtype(positive_df[col]):
                negative_df[col] = 0
            else:
                negative_df[col] = np.nan
    
    return negative_df

# 示例用法
if __name__ == "__main__":   
    
    pos = pd.read_csv("BL--network.csv")
    pos["label"] = 1
    pos.columns=["TF","Target","label"]
    # 方法1: 使用从正样本中提取的基因列表
    negative_df1 = create_negative_samples(pos)
    negative_df1["label"]=0
    negative_df1.columns=["TF","Target","label"]
    print(len(set(list(negative_df1["TF"]))))
    print(len(set(list(pos["TF"]))))
    print((set(list(pos["TF"]))))
    all=pd.concat([pos,negative_df1])   
    print(all)
    all.to_csv("AllPair.csv",index=False)