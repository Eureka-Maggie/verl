import pandas as pd

path = "/primus_xpfs_workspace_T04/txy/data/DAPO-Math-17k/data/dapo-math-17k.parquet"

df = pd.read_parquet(path)

prefix = "Please reason step by step, and put your final answer within \\boxed{{}}.\n Think briefly and concisely - use minimal steps while keeping your reasoning clear.\n\n"
suffix_to_remove = "\n\nRemember to put your answer on its own line after \"Answer:\"."
new_suffix = "\n\n" + prefix.strip()

count = 0

def fix_prompt(messages):
    global count
    result = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Remove prefix that was added in the previous run
            if content.startswith(prefix):
                content = content[len(prefix):]
            # Remove trailing reminder line
            content = content.replace(suffix_to_remove, "")
            # Append new suffix
            content = content + new_suffix
            msg["content"] = content
            count += 1
        result.append(msg)
    return result

df["prompt"] = df["prompt"].apply(fix_prompt)

print(f"Replaced {count} messages")

df.to_parquet(path, index=False)
print("Saved.")
