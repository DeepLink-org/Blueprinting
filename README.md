# Blueprinting

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Blueprinting** is a simulation and analysis toolkit for heterogeneous computing and large-scale distributed training systems. It enables performance modeling, parameter optimization, and bottleneck analysis for LLM training workloads.

[English](#features) | [中文](#功能特性)

---

## Features

- **Performance Simulation**: Model distributed training performance with fine-grained layer-level analysis
- **Distributed Strategy Analysis**: Evaluate DP/TP/PP parallelism strategies and their trade-offs
- **Hardware-Software Co-design**: Explore the design space across model architecture, training configuration, and hardware specifications
- **Numerical Precision Analysis**: Visualize and analyze floating-point formats (FP8, BF16, FP16) for training stability
- **Interactive Web UI**: Streamlit-based dashboard for intuitive exploration and visualization

## Use Cases

| Role | Applications |
|------|-------------|
| **ML Engineers** | Model architecture design, training strategy optimization |
| **Infrastructure Engineers** | Training configuration tuning, performance bottleneck analysis |
| **System Architects** | Network topology evaluation, system configuration optimization |
| **Hardware Designers** | Resource allocation analysis, hardware-model co-optimization |

## Installation

### From PyPI (Coming Soon)

```bash
pip install blueprinting
```

### From Source

```bash
git clone https://github.com/reiase/blueprinting.git
cd blueprinting
pip install -e .
```

### Dependencies

Core dependencies are automatically installed. For the full experience including the web UI:

```bash
pip install -e ".[full]"
```

## Quick Start

### Web Interface

Launch the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard provides:
- **LLM Calculator**: Performance overview, block-level analysis, distributed experiments
- **Precision Analysis**: Floating-point format visualization and comparison

### Command Line Interface

```bash
# Show available commands
blueprinting --help

# Analyze LLM training performance
blueprinting train --model <model.json> --execution <execution.json> --system <system.json>
```

### Python API

```python
import blueprinting as bp

# Define model configuration
model = bp.Model({
    "hidden": 4096,
    "num_blocks": 32,
    "num_heads": 32,
    # ...
})

# Define system configuration
system = bp.System({
    "processor": {"peak_flops": 312e12},
    "memory": {"capacity": 80e9},
    # ...
})

# Define execution parameters
execution = bp.Execution({
    "micro_batch_size": 4,
    "tensor_parallel": 8,
    "pipeline_parallel": 4,
    # ...
})
```

## Project Structure

```
blueprinting/
├── src/
│   ├── blueprinting/      # Core library
│   │   ├── ir/            # Intermediate representation & compiler
│   │   ├── nn/            # Neural network abstractions
│   │   ├── types/         # Type definitions (Model, System, Execution)
│   │   └── fp/            # Floating-point analysis tools
│   ├── calculon/          # Performance calculation engine
│   └── simfloat/          # Floating-point simulation
├── pages/                 # Streamlit UI pages
├── data/                  # Model & system configurations
│   ├── models/            # Pre-defined model configs (GPT, LLaMA, Qwen, etc.)
│   └── systems/           # Hardware system configs (A100, H100, etc.)
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Configuration Examples

Pre-configured model and system files are available in the `data/` directory:

**Models**: GPT-3 (13B, 175B), LLaMA 2/3 (7B-405B), Qwen2 (0.5B-72B), and more

**Systems**: NVIDIA A100, H100 configurations

## Development

### Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format src/
ruff check src/ --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=blueprinting

# Run specific test file
pytest tests/ir/test_compiler.py
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/overview.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Calculon](https://github.com/calculon-ai/calculon) - Performance modeling foundations
- The open-source LLM community for model configurations and validation data

## Citation

If you use Blueprinting in your research, please cite:

```bibtex
@software{blueprinting2024,
  title = {Blueprinting: Heterogeneous Computing and Distributed Training Simulator},
  author = {Reiase},
  year = {2024},
  url = {https://github.com/reiase/blueprinting}
}
```

---

## 功能特性

- **性能仿真**：支持细粒度的分布式训练性能建模
- **分布式策略分析**：评估 DP/TP/PP 并行策略及其权衡
- **软硬协同设计**：探索模型架构、训练配置和硬件规格的设计空间
- **数值精度分析**：可视化分析 FP8、BF16、FP16 等浮点格式对训练稳定性的影响
- **交互式界面**：基于 Streamlit 的可视化仪表板

## 快速开始

### 启动 Web 界面

```bash
streamlit run streamlit_app.py
```

### 命令行使用

```bash
blueprinting --help
```

### 开发模式

```bash
# 开发模式安装
pip install -e .

# 运行测试
pytest

# 代码格式化
ruff format src/
```

## 路线图

- [ ] **核心功能**
  - [ ] P0: 分布式训练仿真
  - [ ] P0: 系统参数寻优
  - [ ] P1: 自动并行优化
  - [ ] P1: 软硬协同设计
- [ ] **用户接口**
  - [ ] P0: Web UI 界面
  - [ ] P1: 命令行界面
  - [ ] P1: Python API
- [ ] **仿真内核**
  - [ ] Transformer 主干结构
  - [ ] MoE 主干结构
  - [ ] DP/TP/PP 并行校验
