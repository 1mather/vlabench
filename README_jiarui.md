# VLABench 使用指南

## 1. 安装

请参考 README.md 完成安装流程。要安装assest。
**注意**：安装依赖时请使用 `requirement1.txt`。

如果assest下载网速慢，可直接拉去我本地的仓库
/mnt/data/310_jiarui/VLABench
tyj@10.10.245.73
gjrgjrhuman
## 2. 测评

### 前置条件
在开始测评前，请先在 LeRobot 上启动 server policy 服务器：(最关键，请参照lerobot/script/server)
- 地址：127.0.0.1
- 端口：8000

### 运行测评
测评文件路径：/mnt/data/310_jiarui/VLABench/tutorials/evaluation.py

注意：如果无法连接server请关闭代理
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

set HTTP_PROXY
set HTTPS_PROXY
set http_proxy
set https_proxy
