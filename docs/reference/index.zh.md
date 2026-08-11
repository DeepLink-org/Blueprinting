# 代码文档

本节使用 `mkdocstrings` 直接从 canonical Python 源码生成，是语义设计文档的 API 伴随视图，不定义第二套 IR 含义。

## 阅读顺序

1. 在 **Canonical IR API** 中选择表示层，查看该层真实 schema 定义。
2. 阅读 **Pass 公式与 API**，理解每个已提交跨层推导的公式、假设与论文来源。
3. 在 **推导基础设施 API** 中查看 decorator、事务 runner、lineage gate 和 deterministic replay contract。

每个公开 canonical Pass 都从类 docstring 生成文档。公式、推导假设、研究来源和明确的非声明与代码放在一起，避免源码 review 和站点文档悄悄描述不同算法。

## 权威边界

- `ir.py` 是 immutable schema 与结构 invariant 的事实源。
- `passes.py` 是公开 Pass contract 的事实源。
- dialect 下的 `*_derivation.py` 保存纯领域推导。
- 渲染页面负责解释和链接，不建立平行 schema 或 estimator。

数学公式由 Arithmatex 与 MathJax 渲染；API 对象和源码清单在 `mkdocs build` 期间由 Python handler 直接从 `src/` 提取。

