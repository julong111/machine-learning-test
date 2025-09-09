# 大米品种二元分类项目总结

## 1. 项目概述

本项目旨在构建一个能够根据大米的物理特征（如面积、周长、偏心率等）准确区分其品种（Cammeo 或 Osmancik）的机器学习模型。这是一个典型的二元分类问题，我们采用 Keras 构建了一个逻辑回归模型来解决它。

与线性回归项目不同，本项目的重点在于展示如何为一个分类任务建立一个标准、清晰、可复用的端到端工作流。我们利用了在之前项目中重构出的 `BaseDataWrapper` 和 `BaseTrainer` 抽象基类，创建了专门用于此任务的 `RiceDataWrapper` 和 `LogisticRegressionTrainer`，从而高效地完成了从数据处理到模型训练、评估和保存的全过程。

## 2. 核心工作流程

本项目的策略遵循了一个非常标准和清晰的机器学习流程：

1.  **数据封装与预处理**: 我们创建了 `RiceDataWrapper`，它继承自通用的 `BaseDataWrapper`。所有特定于大米数据集的处理逻辑，如选择相关特征、对数值特征进行归一化、以及将文本标签（'Cammeo', 'Osmancik'）转换为布尔值（0 或 1）以供模型使用，都被封装在了这个类中。

2.  **训练器封装**: 我们创建了 `LogisticRegressionTrainer`，它继承自通用的 `BaseTrainer`。所有与模型训练相关的逻辑，包括构建二元分类模型、在训练集上进行训练、在测试集上进行评估、以及保存所有相关工件（模型、配置、图表），都被封装在了这个类中。

3.  **主流程编排**: `rice_training.py` 作为项目的主入口，负责编排整个流程。它解析命令行参数，调用 `RiceDataWrapper` 准备数据，然后实例化并运行 `LogisticRegressionTrainer` 来完成整个机器学习生命周期。

## 3. 各脚本详细解析

以下是项目中各个核心脚本的作用与流程的详细解析。

---

### **`src/binary_classification_rice_2/rice_data_wrapper.py` (数据处理核心)**

*   **核心作用**: 这是本项目的数据处理中枢。它继承自 `BaseDataWrapper`，负责所有与大米数据集相关的加载、预处理和分割工作。

*   **处理流程**:
    1.  **继承与初始化**: 调用 `BaseDataWrapper` 的 `__init__`。
    2.  **实现 `_preprocess`**: 这是最核心的特定逻辑。它首先从原始数据中挑选出与分类任务最相关的物理特征列，然后调用 `stats_utils.normalize_features` 对这些数值特征进行归一化，最后，它将 'Class' 列中的文本标签 'Cammeo' 转换为数值 `1`，另一个品种转换为 `0`，并创建一个新的 `Class_Bool` 列作为模型的目标标签。
    3.  **实现 `_get_final_columns`**: 定义模型训练最终需要的列，包括输入特征和 `Class_Bool` 标签。
    4.  **实现 `_get_dataset_name`**: 返回 "rice"，用于生成统一命名的数据文件（`rice_train.csv` 等）。
    5.  **实现 `_get_split_args`**: 指定了数据分割的参数，本项目中使用 64/16/20 的比例来划分训练、验证和测试集。

*   **关键算法/方法**:
    *   **特征归一化 (`stats_utils.normalize_features`)**: 将所有数值特征缩放到相似的范围，这对于梯度下降类的模型（如本项目的逻辑回归）至关重要，可以帮助模型更快、更稳定地收敛。

---

### **`src/binary_classification_rice_2/logistic_regression_trainer.py` (模型训练核心)**

*   **核心作用**: 封装了逻辑回归模型的所有操作，继承自 `BaseTrainer`。

*   **处理流程**:
    1.  **继承与初始化**: 调用 `BaseTrainer` 的 `__init__`，并传入固定的标签列名 `Class_Bool`。
    2.  **实现 `build_model`**: 调用通用的 `keras_utils.create_binary_classification_model` 来构建一个 Keras 模型。这个模型包含一个输入层、一个或多个隐藏层（可选），以及一个使用 `sigmoid` 激活函数的输出层，专门用于二元分类。
    3.  **实现 `evaluate`**: 在测试集上评估模型性能。它不仅会打印出标准的分类指标（**Loss, Accuracy, Precision, Recall, AUC**），还会利用 `prediction_utils` 展示一部分样本的预测概率和最终预测类别，以便进行直观的分析。
    4.  **实现 `_plot_history`**: 调用 `plot_utils.plot_classification_history`，将训练过程中的准确率、AUC等关键指标的变化曲线绘制成图并保存。

---

### **`src/binary_classification_rice_2/rice_training.py` (主训练脚本)**

*   **核心作用**: 项目的统一入口，负责编排数据处理和模型训练的整个流程。

*   **处理流程**:
    1.  **参数解析**: 解析命令行参数（尽管本项目只有一个默认模型 `rice_model`）。
    2.  **路径与配置设置**: 定义所有工件（数据、模型、图表）的存放路径。
    3.  **数据加载/处理**: 检查是否存在已处理好的数据文件。如果不存在，则实例化 `RiceDataWrapper` 并调用其 `process_and_split` 方法来生成它们。
    4.  **训练器驱动**: 实例化 `LogisticRegressionTrainer`，然后依次调用其 `build_model()`, `train()`, `evaluate()`, 和 `save()` 方法，以声明式的方式完成整个工作流。

*   **关键参数与选择原因**:
    *   `learning_rate=0.001`, `number_epochs=60`, `batch_size=100`: 这是一组在分类任务中表现稳健的超参数。`0.001` 的学习率较为通用，`60` 个周期确保模型有足够的时间来学习，而 `100` 的批次大小则在训练速度和梯度稳定性之间取得了平衡。

---

## 4. 项目总结与展望

本项目成功地演示了如何利用一个可复用的代码框架，快速、清晰地构建一个端到端的二元分类项目。通过继承 `BaseDataWrapper` 和 `BaseTrainer`，我们得以将精力完全集中在特定于任务的数据处理和模型评估逻辑上，极大地提高了开发效率和代码的可维护性。

### **下一步展望**

当前模型已经取得了不错的性能，但仍有许多可以探索的优化方向：

#### **1. 深度特征分析与选择**

*   **特征重要性分析**: 使用像 `scikit-learn` 中的 `SelectKBest` 或基于模型的特征选择（如从梯度提升树中获取特征重要性），来评估当前所有特征对分类的贡献度。这可能会发现某些特征是冗余的，或者某些特征的组合更有价值。

#### **2. 尝试更强大的模型**

*   **梯度提升树 (XGBoost, LightGBM)**: 对于表格数据分类任务，梯度提升模型通常是性能的“天花板”。它们能够自动学习特征之间的复杂交互和非线性关系，很可能在当前数据集上取得比逻辑回归更高的准确率。
*   **更深或更宽的神经网络**: 我们可以调整 `keras_utils` 中的模型构建函数，尝试增加更多的隐藏层或神经元，以增强模型的拟合能力，但需要注意过拟合的风险。

#### **3. 超参数调优**

*   **系统性搜索**: 使用 `GridSearchCV` 或 `RandomizedSearchCV` 等工具，对学习率、批次大小、周期数甚至优化器类型（如 Adam, RMSprop）进行系统性的搜索，以找到最优的超参数组合。

#### **4. 处理类别不平衡 (如果存在)**

*   **检查与应对**: 首先需要检查数据集中 'Cammeo' 和 'Osmancik' 两个类别的样本数量是否大致相等。如果存在显著的不平衡，可能会导致模型偏向于多数类。可以采用以下方法应对：
    *   **类别权重 (Class Weights)**: 在模型训练时，为样本量较少的类别赋予更高的权重，使得模型在计算损失时更加关注少数类的错误。
    *   **重采样 (Resampling)**: 通过对少数类进行过采样（如 SMOTE 算法）或对多数类进行欠采样，来人为地平衡数据集。

## 5. 脚本执行命令指南

为了方便回顾和复现，以下是本项目核心脚本的推荐执行顺序和命令。请在项目根目录 (`machine-learning-test`) 下运行。

```bash
# 步骤 1: 训练、评估并保存模型
# 该脚本会自动处理数据准备（如果需要）和整个模型生命周期
python -m src.binary_classification_rice_2.rice_training

# (可选) 如果想重新强制生成数据，可以先删除 artifacts/binary_classification_rice_2 目录下的
# rice_train.csv, rice_validation.csv, 和 rice_test.csv 文件，然后再次运行上面的命令。

# 步骤 2: 在测试集上重新评估已保存的模型
# (注意: 当前的 rice_training.py 已经包含了评估步骤，
# test.py 脚本主要是为了演示如何独立加载和评估一个已存在的模型)
python -m src.binary_classification_rice_2.test rice_model
```
