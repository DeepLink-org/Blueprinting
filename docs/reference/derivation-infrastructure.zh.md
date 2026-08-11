# 推导基础设施 API

基础设施分离三个关注点：

- schema authoring 创建 immutable、codec-visible 的 record 与封闭 ADT；
- pass authoring 从 annotation 提取静态 contract，不包装执行语义；
- transaction runner 在 commit 前验证输入、输出、lineage rule、deterministic replay、analysis 与 checkpoint。

## Schema authoring

::: blueprinting.schema.authoring
    options:
      members:
        - record
        - adt
        - variant
        - adt_manifest
      show_root_heading: false
      show_root_toc_entry: false

## Pass authoring

::: blueprinting.synthesizer.passes.authoring
    options:
      members:
        - derivation
        - relation
        - claim
      show_root_heading: false
      show_root_toc_entry: false

## 事务与验证类型

::: blueprinting.synthesizer.passes.base
    options:
      members:
        - PassRule
        - TransitionRelation
        - TransitionReport
        - TransitionVerifier
        - PassContract
        - PassContext
        - PassResult
        - DerivationPass
        - PassPipeline
        - PassRecord
        - PassCheckpoint
        - PassManager
      show_root_heading: false
      show_root_toc_entry: false
