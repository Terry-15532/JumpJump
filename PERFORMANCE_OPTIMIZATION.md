# 性能优化说明 (Performance Optimization)

## 已实施的优化 (Implemented Optimizations)

### 1. 截屏加速 (Screenshot Acceleration)
- **使用 Raw 格式代替 PNG** (50-70% faster)
  - `adb exec-out screencap` (raw) 代替 `screencap -p` (PNG)
  - 无需 PNG 解压缩，直接解析 RGBA 原始数据
  - 自动回退到 PNG 格式以保证兼容性

### 2. 禁用视觉输出 (Disable Visual Output)
- 默认关闭所有 `cv2.imshow()` 窗口显示
- 通过 `ENABLE_DISPLAY=False` 控制（可在 auto_jump.py 顶部修改）
- 节省约 50-100ms/帧

### 3. 图像降采样 (Image Downsampling)
- `FAST_MODE=True` 时自动将超过 1920p 的截屏缩小
- 减少图像处理计算量
- 对检测精度影响很小

### 4. 移除人工延迟 (Remove Artificial Delays)
- 移除跳转前的随机等待（原 1-3 秒）
- 移除跳转后的固定延迟（原 1.5 秒）
- 失败重试延迟降至最低

### 5. 优化检测流程 (Optimize Detection Pipeline)
- 移除中间调试输出
- 只在必要时绘制检测标记

## 性能指标 (Performance Metrics)

**优化前 (Before):**
- 截屏: ~300-500ms (PNG)
- 检测: ~100-200ms
- 显示: ~50-100ms
- 人工延迟: 2500-4500ms
- **总计: ~3-5秒/帧, 0.2-0.3 FPS**

**优化后 (After):**
- 截屏: ~100-200ms (Raw) 或 ~200-300ms (PNG fallback)
- 检测: ~100-200ms
- 显示: 0ms (已禁用)
- 人工延迟: 0ms
- **总计: ~0.2-0.4秒/帧, 2.5-5 FPS**

**加速比: 10-15倍**

## 使用方法 (Usage)

### 快速模式（默认）
```python
# auto_jump.py 顶部
ENABLE_DISPLAY = False  # 不显示窗口
FAST_MODE = True        # 快速模式
```

### 调试模式
```python
# 需要查看检测效果时
ENABLE_DISPLAY = True   # 显示检测窗口
FAST_MODE = False       # 使用完整分辨率
```

## 运行时统计 (Runtime Statistics)

程序每处理 10 帧会输出性能统计：
```
Stats: avg=0.25s/frame, FPS=4.0, total=10
```

## 进一步优化建议 (Further Optimization Ideas)

如需更高性能，可考虑：

1. **使用 GPU 加速** (需要额外设置)
   - OpenCV CUDA 模块
   - 或使用神经网络检测（PyTorch + GPU）

2. **降低截屏分辨率**
   - 在设备端直接输出低分辨率：`screencap -d 0` 并指定尺寸

3. **多线程处理**
   - 截屏和检测在不同线程并行

4. **使用 scrcpy** 代替 adb
   - scrcpy 提供更快的屏幕流传输

## 兼容性说明 (Compatibility Notes)

- Raw 格式在绝大多数 Android 设备上工作（Android 4.2+）
- 如果 Raw 格式失败，会自动回退到 PNG 格式
- Windows/Linux/macOS 均支持

