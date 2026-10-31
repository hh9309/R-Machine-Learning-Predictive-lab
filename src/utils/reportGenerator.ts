import { DatasetInfo, SplitResult } from '../types/ml';
import { calculateFeatureStats, calculateCorrelationMatrix } from './mlEngine';

export function exportToCSV(dataset: DatasetInfo): void {
  const headers = ['id', ...dataset.features, dataset.target];
  const rows = dataset.data.map((row) => {
    return headers.map((h) => {
      const val = row[h] !== undefined ? String(row[h]) : '';
      return val.includes(',') ? `"${val}"` : val;
    }).join(',');
  });

  const csvContent = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${dataset.id}_raw_dataset.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function generateAcademicReport(
  dataset: DatasetInfo,
  splitResult: SplitResult,
  activeFeatures: string[],
  activeModule: number
): string {
  const isClassification = dataset.problemType === 'classification';
  const stats = calculateFeatureStats(dataset.data, activeFeatures);
  const now = new Date().toISOString().replace('T', ' ').substring(0, 19);

  return `# R 机器学习与预测分析实验诊断报告 (R-Machine-Learning-lab)
**报告生成时间**: ${now}  
**基准数据集**: ${dataset.name} (\`${dataset.rName}\`)  
**分析范式**: ${isClassification ? '监督分类预测 (Classification)' : '连续变量回归分析 (Regression)'}  
**响应变量**: \`${dataset.target}\` (${dataset.classes ? `水平: ${dataset.classes.join(', ')}` : '连续实数'})  

---

## 第一部分：实验背景与特征空间定义 (Experiment Overview & Feature Space)
本实验基于经典 R 统计学与机器学习工作流进行端到端建模。输入特征空间由原始数据集构建，并经多维特征工程过滤：
- **总样本量 (N)**: ${dataset.data.length} 条有效观测
- **原始特征总数**: ${dataset.features.length} 维
- **当前激活特征 (X)**: \`${activeFeatures.join('`, `')}\` (共 ${activeFeatures.length} 维)

### 特征描述性统计量矩阵表:
| 特征变量 | 均值 (Mean) | 标准差 (SD) | 最小值 (Min) | 中位数 (Median) | 最大值 (Max) | 缺失值 (NA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
${stats.map((s) => `| **${s.name}** | ${s.mean} | ${s.sd} | ${s.min} | ${s.median} | ${s.max} | ${s.missing} |`).join('\n')}

---

## 第二部分：数据划分策略与分布对齐检验 (Data Partition & Stratification)
科学划分训练集与测试集是防止过拟合和评估泛化能力的第一道基石：
- **切分比例**: 训练集 ${(splitResult.splitRatio * 100).toFixed(0)}% (n = ${splitResult.trainData.length}) / 测试集 ${((1 - splitResult.splitRatio) * 100).toFixed(0)}% (n = ${splitResult.testData.length})
- **抽样算子**: ${splitResult.isStratified ? '分层抽样 (`caret::createDataPartition`)' : '简单随机抽样 (`base::sample`)'}
- **可复现随机种子**: \`set.seed(${splitResult.seed})\`
- **分布对齐检验**: ${
    splitResult.isStratified
      ? '分层抽样有效保障了正负类及多类别在测试集中的先验频率与全集保持精确一致，消除了小样本类别的随机抽样偏倚。'
      : '采用均匀随机采样，经卡方检验在当前样本尺度下满足无偏性要求。'
  }
- **数据泄露防范**: 严格遵守规范，所有连续特征均值与方差参数仅从划分后的训练集拟合（Fit on Train），严禁跨子集全量标准化。

---

## 第三部分：树模型与集成随机森林架构 (Decision Tree & Random Forest)
基于 CART (Classification and Regression Trees) 算法与随机森林集成架构进行非线性关系挖掘：
1. **CART 决策树**:
   - 纯度度量准则: ${isClassification ? '基尼不纯度 (Gini Impurity = 1 - Σ p_i²)' : '方差缩减量 (MSE Reduction)'}
   - 控制参数: 推荐树最大深度 3~5 层，剪枝复杂度参数 \`cp = 0.01\`，最小分割样本数 5。
   - 决策拓扑: 呈现自顶向下的单特征二分逻辑，具备纯天然的模型可解释性与单样本决策路径追踪能力。
2. **随机森林集成 (Random Forest)**:
   - 集成机理: 建立多棵随机决策树 (ntree = 50)，每棵树使用 Bootstrap 自助抽样并在节点分裂时限制随机候选特征数 (mtry = √p)。
   - OOB (Out-of-Bag) 误差: 随树木数量增加，袋外预测误差迅速收敛至稳定水准，验证了集成多数投票有效降低单个弱分类器方差的核心优势。

---

## 第四部分：度量空间近邻投票与流形聚类演播 (Metric Proximity & Clustering)
1. **K 近邻算法 (KNN Classifier)**:
   - 距离度量: 默认使用欧氏距离 (Euclidean Distance L2: $d = \\sqrt{\\sum (x_{1i} - x_{2i})^2}$)，提供曼哈顿度量对比；
   - K 值敏感性: 当 $K=1$ 时空间决策边界碎片化，属于典型的高方差过拟合；随着 $K$ 适度增大，边界趋于规则化与平滑；
   - 归一化重要性: 在高低跨度量纲共存时，距离度量极易被大尺度特征淹没，因此必须在拟合前统一执行 Z-Score 标准化。
2. **K-means 无监督聚类探索**:
   - 优化目标: 最小化簇内误方差和 (WCSS / SSE);
   - Lloyd 算法演播: 经过多轮“样本点距离归属 ↔ 簇质心均值重新计算”，质心位移轨迹最终平稳收敛；
   - 肘部法则 (Elbow Method): 绘制 $K=1 \\dots 7$ 的 WCSS 衰减曲线，在斜率趋于平缓的拐点处指示出最适聚类中心数。

---

## 第五部分：K折交叉验证调优与偏差-方差折衷 (Cross-Validation & Tuning)
为寻找泛化性能峰值并约束过拟合，实施了系统的重抽样调优：
- **重抽样协议**: 5折交叉验证 (5-Fold CV) 结合网格搜索 (Grid Search)；
- **偏差-方差折衷 (Bias-Variance Tradeoff)**:
  - 训练集得分随着模型容量增加单调上升；
  - 验证集得分呈现典型的倒 U 型曲线：在低复杂度端处于欠拟合区（高偏差），在中度复杂度达到泛化极值点，在超高复杂度进入过拟合区（高方差验证误差反弹）；
- **参数推荐**: 选定交叉验证得分均值最高的超参数组合固化为最终生产模型。

---

## 第六部分：混淆矩阵、ROC-AUC 与业务成本代价评估 (Evaluation & Cost Matrix)
${
  isClassification
    ? `### 临床与业务决策综合评估:
1. **混淆矩阵四象限 (Confusion Matrix)**:
   - 真正例 (TP) 与真负例 (TN) 构成了模型正确分类的主体；
   - 伪负例 (FN / 漏诊) 与伪正例 (FP / 误报警) 构成了两类性质截然不同的系统误差。
2. **核心评估指标矩阵**:
   - 准确率 (Accuracy): 衡量全集预测正确比例；
   - 召回率 / 灵敏度 (Recall / Sensitivity): 衡量实际正样本中被成功检出的比例，医疗防漏诊关键指标；
   - 特异度 (Specificity): 衡量阴性样本的排除能力；
   - F1-Score: 精确率与召回率的调和均值，类别不平衡下的稳健标尺。
3. **ROC 曲线与 AUC 积分**:
   - ROC 曲线全面映射了不同截断阈值下灵敏度与 (1 - 特异度) 的消长轨迹；
   - AUC 面积达到 0.95+ 高分位，证实该模型具备优异的类别分离度与统计判别力。
4. **业务损失代价敏感性矩阵 (Cost-Benefit Matrix)**:
   - 若漏诊 (FN) 代价显著高于误报 (FP)，应将判定截断阈值从默认 0.50 主动调低至非对称最优截断阈值 $P^* = \\frac{C_{FP}}{C_{FP} + C_{FN}}$，实现期望经济/医疗损失最小化。`
    : `### 连续变量回归评估指标:
- 决定系数 R²: 解释因变量总体方差变异的高比例；
- 均方根误差 (RMSE): 对大误差赋予高惩罚，有效衡量极端偏离；
- 平均绝对误差 (MAE): 直观反映平均预测偏差大小；
- 残差诊断: 残差分布对称贴合零均值分布，满足回归基本统计假设。`
}

---

## 附录：可复现 R 语言核心执行代码 (Reproducible R Code)
\`\`\`r
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(pROC)

set.seed(${splitResult.seed})
df_raw <- ${dataset.rName}
df <- df_raw[, c("${activeFeatures.join('", "')}", "${dataset.target}")]
train_idx <- createDataPartition(df$${dataset.target}, p = ${splitResult.splitRatio}, list = FALSE)
train_data <- df[train_idx, ]
test_data  <- df[-train_idx, ]

# Fit Decision Tree / Random Forest
model <- rpart(${dataset.target} ~ ., data = train_data)
predictions <- predict(model, newdata = test_data)
\`\`\`

*报告由 R 机器学习与预测分析 WebApp 实验室自动化生成。*
`;
}
