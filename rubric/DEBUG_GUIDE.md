# RL训练调试指南 - 如何打断点和停住程序

## 方法1: 使用Python内置的pdb调试器（推荐）

### 在关键位置插入断点

在你想停住的地方插入以下代码：

```python
import pdb; pdb.set_trace()
```

### 关键断点位置

#### 1. 在rubric_generator.py中打断点

```python
# 在 sample_rollouts 方法开始处
def sample_rollouts(self, batch: DataProto) -> List[Dict]:
    import pdb; pdb.set_trace()  # 断点1: 查看batch数据
    prompts = batch.batch["prompts"]
    ...

# 在格式化样本后
def generate_rubric(self, batch: DataProto, step: int = 1):
    ...
    formatted_samples = self.format_rollout_samples(sampled_rollouts)
    import pdb; pdb.set_trace()  # 断点2: 查看格式化后的样本

    # 在调用LLM之前
    prompt_with_samples = self.rubric_template.replace(...)
    import pdb; pdb.set_trace()  # 断点3: 查看完整的prompt
    response, success = self.llm_client.simple_text_call(...)
```

#### 2. 在ray_trainer.py中打断点

```python
# 在第一次rollout后触发rubric生成的地方（约1611行）
if self.global_steps == 1 and hasattr(self, "rubric_generator"):
    import pdb; pdb.set_trace()  # 断点4: 第一次rollout完成
    rubric_path = self.rubric_generator.generate_rubric(batch, step=self.global_steps)
```

### pdb调试命令

当程序停在断点时，你可以使用这些命令：

```bash
# 查看变量
p variable_name              # 打印变量值
pp variable_name             # 漂亮打印变量
type(variable_name)          # 查看变量类型

# 查看数据结构
p batch.batch.keys()         # 查看batch中有哪些key
p len(prompts)               # 查看prompts数量
p prompts[0]                 # 查看第一个prompt
p formatted_samples[:500]    # 查看前500个字符

# 执行代码
!import torch                # 执行任意Python代码（加!前缀）
!print(batch.batch["prompts"].shape)

# 继续执行
n                            # 下一行（next）
s                            # 进入函数（step into）
c                            # 继续执行到下一个断点（continue）
q                            # 退出调试（quit）

# 查看代码
l                            # 列出当前位置的代码
ll                           # 列出整个函数的代码
w                            # 查看调用栈（where）
```

## 方法2: 使用ipdb（增强版pdb，推荐）

### 安装ipdb

```bash
conda activate /primus_xpfs_workspace_T04/txy/venv/verl
pip install ipdb
```

### 使用ipdb断点

```python
import ipdb; ipdb.set_trace()  # 比pdb更友好，有语法高亮和自动补全
```

## 方法3: 使用条件断点

只在特定条件下停住：

```python
# 只在第一次rollout时停住
if self.global_steps == 1:
    import pdb; pdb.set_trace()

# 只在采样到特定数量的rollout时停住
if len(sampled_rollouts) >= 16:
    import pdb; pdb.set_trace()
```

## 方法4: 使用VSCode远程调试（最强大）

### 1. 在VSCode中配置launch.json

创建 `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/primus_xpfs_workspace_T04/txy/projects/verl"
                }
            ]
        }
    ]
}
```

### 2. 在代码中添加调试服务器

```python
# 在 rubric_generator.py 或 ray_trainer.py 开头添加
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()  # 程序会在这里等待VSCode连接
print("Debugger attached!")
```

### 3. 运行训练并连接调试器

```bash
# 运行训练脚本
bash examples/grpo_trainer/run_qwen3_4b_grpo_dy.sh

# 在VSCode中按F5或点击"Run and Debug" -> "Python: Attach"
```

## 实际操作步骤

### 快速开始：在第一次rollout后停住

1. **编辑 ray_trainer.py**

```bash
vim /primus_xpfs_workspace_T04/txy/projects/verl/verl/trainer/ppo/ray_trainer.py
```

在约1611行添加断点：

```python
# Generate rubric after first rollout if enabled
if self.global_steps == 1 and hasattr(self, "rubric_generator") and self.rubric_generator is not None:
    import pdb; pdb.set_trace()  # 🔴 断点：第一次rollout完成
    try:
        print("\n" + "=" * 60)
        print("Generating rubric after first rollout...")
        ...
```

2. **编辑 rubric_generator.py**

```bash
vim /primus_xpfs_workspace_T04/txy/projects/verl/verl/utils/rubric_generator.py
```

在关键位置添加断点：

```python
def sample_rollouts(self, batch: DataProto) -> List[Dict]:
    import pdb; pdb.set_trace()  # 🔴 断点：查看batch数据
    prompts = batch.batch["prompts"]
    ...

def generate_rubric(self, batch: DataProto, step: int = 1):
    ...
    prompt_with_samples = self.rubric_template.replace(...)
    import pdb; pdb.set_trace()  # 🔴 断点：查看完整prompt
    response, success = self.llm_client.simple_text_call(...)
```

3. **运行训练**

```bash
cd /primus_xpfs_workspace_T04/txy/projects/verl
bash examples/grpo_trainer/run_qwen3_4b_grpo_dy.sh
```

4. **当程序停在断点时**

```python
# 查看batch数据
(Pdb) p batch.batch.keys()
dict_keys(['prompts', 'responses', 'token_level_rewards', ...])

# 查看prompts数量
(Pdb) p len(batch.batch["prompts"])
512

# 查看第一个prompt（解码后）
(Pdb) p self.tokenizer.decode(batch.batch["prompts"][0])
'Solve the equation: 2x + 3 = 7'

# 查看rewards
(Pdb) p batch.batch["token_level_rewards"][0].mean()
tensor(0.8500)

# 继续执行到下一个断点
(Pdb) c
```

## 常见调试场景

### 场景1: 查看采样的rollout内容

```python
# 在 sample_rollouts 返回前
import pdb; pdb.set_trace()
# 然后在pdb中：
(Pdb) pp sampled_rollouts[0]
{
    'prompt': 'Solve the equation...',
    'response': 'Let me solve this...',
    'score': 0.85
}
```

### 场景2: 查看发送给LLM的完整prompt

```python
# 在调用LLM前
import pdb; pdb.set_trace()
# 然后：
(Pdb) print(prompt_with_samples)
# 或保存到文件
(Pdb) !with open('/tmp/prompt.txt', 'w') as f: f.write(prompt_with_samples)
```

### 场景3: 查看LLM返回的response

```python
# 在LLM调用后
response, success = self.llm_client.simple_text_call(...)
import pdb; pdb.set_trace()
# 然后：
(Pdb) print(response)
(Pdb) !with open('/tmp/response.txt', 'w') as f: f.write(response)
```

## 不停住程序，只输出到日志文件

如果你不想停住程序，只想查看中间结果：

```python
# 在 rubric_generator.py 中
def generate_rubric(self, batch: DataProto, step: int = 1):
    # 保存完整prompt到文件
    debug_dir = Path("/tmp/rubric_debug")
    debug_dir.mkdir(exist_ok=True)

    with open(debug_dir / f"prompt_step_{step}.txt", "w") as f:
        f.write(prompt_with_samples)

    # 调用LLM
    response, success = self.llm_client.simple_text_call(...)

    # 保存response到文件
    with open(debug_dir / f"response_step_{step}.txt", "w") as f:
        f.write(response)

    print(f"Debug files saved to {debug_dir}")
```

## 总结

**最简单的方法**：在想停住的地方加 `import pdb; pdb.set_trace()`

**最推荐的位置**：
1. `ray_trainer.py:1611` - 第一次rollout完成时
2. `rubric_generator.py:sample_rollouts` 开始 - 查看batch数据
3. `rubric_generator.py:generate_rubric` LLM调用前 - 查看完整prompt
4. `rubric_generator.py:generate_rubric` LLM调用后 - 查看response

**运行后**：程序会停在断点，你可以用pdb命令查看任何变量，然后按 `c` 继续执行。
