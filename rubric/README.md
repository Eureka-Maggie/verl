# Automatic Rubric Generation for RL Training

This implementation adds automatic rubric generation after the first rollout in verl training.

## Overview

After the first rollout completes, the system will:
1. Randomly select 8 prompts from the rollout data
2. For each prompt, select 2 rollouts (preferably one correct and one incorrect)
3. Insert these samples into the rubric template
4. Call an LLM to generate evaluation rubrics
5. Save the generated rubrics to `{output_dir}/{exp_name}/step_1.jsonl`

## Files Created/Modified

### New Files

1. **`verl/utils/llm_client.py`**
   - Encapsulates LLM API calling functionality
   - Supports multiple response formats (Gemini, Qwen, etc.)
   - Handles authentication and error handling

2. **`verl/utils/rubric_generator.py`**
   - Main rubric generation logic
   - Samples rollouts from batch data
   - Formats samples and calls LLM
   - Extracts and saves generated rubrics

3. **`rubric/init_rubric_refined.txt`**
   - Refined rubric generation prompt template
   - Cleaned up from original with Chinese annotations removed

4. **`rubric/config_example.yaml`**
   - Example configuration for enabling rubric generation

5. **`examples/grpo_trainer/run_qwen3_4b_grpo_dy.sh`**
   - Modified training script with rubric generation enabled

### Modified Files

1. **`verl/trainer/ppo/ray_trainer.py`**
   - Added rubric generator initialization in `__init__` method (lines 304-339)
   - Added rubric generation hook after first rollout in `fit` method (lines 1610-1624)

## Usage

### 1. Set up LLM API credentials

Create a `.env` file at `/primus_xpfs_workspace_T04/txy/projects/verl/rubric/.env`:

```bash
LLM_TOKEN=your_token_here
LLM_API_URL=https://llm-chat-api.alibaba-inc.com/v1/api/chat
LLM_MODEL=gemini-3-flash-preview
LLM_APP=quark_gen
LLM_BUSINESS_UNIT=your_business_unit
LLM_QUOTA_ID=your_quota_id
LLM_USER_ID=your_user_id
LLM_ACCESS_KEY=your_access_key
```

### 2. Run training with rubric generation

Use the modified training script:

```bash
bash examples/grpo_trainer/run_qwen3_4b_grpo_dy.sh
```

Or add these parameters to your existing training command:

```bash
python3 -m verl.trainer.main_ppo \
    ... (your existing parameters) \
    trainer.enable_rubric_generation=true \
    trainer.rubric_generation.template_path="/primus_xpfs_workspace_T04/txy/projects/verl/rubric/init_rubric_refined.txt" \
    trainer.rubric_generation.output_dir="/primus_xpfs_workspace_T04/txy/projects/verl/rubric" \
    trainer.rubric_generation.num_prompts=8 \
    trainer.rubric_generation.num_rollouts_per_prompt=2 \
    trainer.rubric_generation.llm_env_path="/primus_xpfs_workspace_T04/txy/projects/verl/rubric/.env"
```

### 3. Check generated rubrics

After the first rollout completes, rubrics will be saved to:
```
/primus_xpfs_workspace_T04/txy/projects/verl/rubric/{exp_name}/step_1.jsonl
```

Each line in the JSONL file contains one rubric in the format:
```json
{
    "rubric_id": "gen_001",
    "title": "Final Answer Presence",
    "principle": "Score 1 iff the response contains an explicit final answer statement, clearly separated from the intermediate reasoning; otherwise 0."
}
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.enable_rubric_generation` | `false` | Enable/disable rubric generation |
| `trainer.rubric_generation.template_path` | `rubric/init_rubric_refined.txt` | Path to rubric template |
| `trainer.rubric_generation.output_dir` | `rubric/` | Output directory for rubrics |
| `trainer.rubric_generation.num_prompts` | `8` | Number of prompts to sample |
| `trainer.rubric_generation.num_rollouts_per_prompt` | `2` | Rollouts per prompt |
| `trainer.rubric_generation.llm_env_path` | `rubric/.env` | Path to LLM credentials |

## Implementation Details

### Rollout Sampling Strategy

The `RubricGenerator.sample_rollouts()` method:
1. Groups rollouts by prompt
2. Randomly selects `num_prompts` unique prompts
3. For each prompt, selects `num_rollouts_per_prompt` rollouts:
   - If 2 rollouts requested: selects lowest and highest scoring (diverse samples)
   - Otherwise: random sampling
4. This ensures a mix of correct and incorrect responses when possible

### LLM Integration

The `LLMClient` class:
- Loads credentials from `.env` file
- Supports multiple response formats
- Handles timeouts and errors gracefully
- Prints cost/usage information when available

### Rubric Extraction

The `extract_rubrics_from_response()` method:
- First tries to parse entire response as JSON
- Falls back to regex extraction of individual JSON objects
- Validates rubric structure (requires `rubric_id`, `title`, `principle`)
- Returns list of valid rubrics

## Troubleshooting

### Rubric generation fails silently

Check the console output for error messages. Common issues:
- Missing or invalid `.env` file
- LLM API credentials incorrect
- Template file not found
- No rollout data available

### No rubrics extracted from LLM response

The LLM response may not contain valid JSON. Check:
- LLM model supports JSON output
- Template prompt is clear about output format
- Response is being printed (check logs)

### Import errors

Ensure you're running in the correct conda environment:
```bash
conda activate /primus_xpfs_workspace_T04/txy/venv/verl
```

## Future Improvements

1. Support for multiple rubric generation steps (not just step 1)
2. Configurable sampling strategies (e.g., stratified by reward)
3. Rubric validation and quality checks
4. Integration with reward model training
5. Support for other LLM providers (OpenAI, Anthropic, etc.)
