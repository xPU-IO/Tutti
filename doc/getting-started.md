# Tutti 快速开始

Tutti 默认采用包含 CUDA、local NVMe、SNVMe 模块、daemon 和硬件测试的完整硬件构建，输出路径为 `build/`。如需使用 MUSA/MACA、调整编译器或维护内核模块，参考 [`advanced-build.md`](advanced-build.md)。

## 构建

```bash
cd /data/home/ryeqiu/Tutti
./scripts/prepare_env.sh
cmake --preset default
cmake --build --preset default --parallel 8
```

`prepare_env.sh` 会先检查系统依赖和仓库中已有的 vcpkg；已满足时直接退出，不会下载或编译 vcpkg。

构建产物：

```text
build/module/snvme-core.ko
build/module/snvme.ko
build/bin/tutti_daemon
```

## 启动

```bash
cmake --build --preset default --target insmod
sudo build/bin/tutti_daemon --config config/local/tutti_daemon.yaml
```

第二条命令使用本机 daemon YAML；配置方式见 [`tutti_daemon.md`](tutti_daemon.md)。

## 测试

```bash
ctest --preset default
```

## 日常重新构建

```bash
cmake --build --preset default --parallel 8
```

## 清理

```bash
rm -rf build
```

KV Cache workload 运行命令在
[`../examples/layerwise_kv_overlap/layerwise_kv_overlap.cpp`](../examples/layerwise_kv_overlap/layerwise_kv_overlap.cpp)
文件头中；其他参数使用 `build/bin/tutti_layerwise_kv_overlap --help` 查看。
