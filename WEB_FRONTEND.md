# PPG Vessel Web Workbench

## 启动

```powershell
python web_frontend.py --host 127.0.0.1 --port 8765
```

浏览器打开:

```text
http://127.0.0.1:8765
```

## 指定 Conda 环境

推荐方式: 编辑 [web_frontend_config.json](E:/PPG_Prediction/web_frontend_config.json) 里的 `conda_env`:

```json
{
  "host": "127.0.0.1",
  "port": 8765,
  "conda_env": "base",
  "conda_exe": ""
}
```

然后运行:

```powershell
.\start_web_frontend.ps1
```

也可以临时覆盖环境:

```powershell
.\start_web_frontend.ps1 -CondaEnv pytorch
```

或直接用 Python:

```powershell
python web_frontend.py --conda-env pytorch
```

如果 `conda` 不在 PATH 中，可以把 `conda_exe` 改成完整路径，例如:

```json
"conda_exe": "D:\\anaconda\\Scripts\\conda.exe"
```

## 功能

- 单文件模式: 上传一个 `.stl`，输出写入 `web_runs/<session>/...`，也可以填写输出目录。
- 批量模式: 输入病例根目录，默认查找每个子目录下的 `vessel.stl`。
- 流程按钮: 中心线提取、平滑、分段、逐点截面特征、统计特征、导出可视化。
- 自动全流程: 按顺序运行全部步骤；批量模式会处理全部病例。
- 3D 图层: STL 模型、原始/平滑中心线、分段、分叉点、曲率特征点、间隔截面、最大截面、平均截面、标签。
- 下载结果: 打包 STL、各中间文件、最终 JSON 和导出 HTML/PNG。

## 依赖

前端服务本身只依赖 Python 标准库和本仓库内已有文件。实际运行算法时仍需要原项目依赖，例如:

```text
numpy scipy scikit-image networkx trimesh shapely plotly vtk kaleido pillow
```

如果依赖缺失，页面中的任务日志会显示对应的 `ModuleNotFoundError` 或算法异常。
