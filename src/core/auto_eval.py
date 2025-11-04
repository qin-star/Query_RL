import pandas as pd

from src.utils.llm import get_chat_llm, SafeParser
from src.utils.prompt import g_sa_prompt_manager

llm = get_chat_llm("glm45fp8-auth")

EXCEL_PATH = '/home/jovyan2/query_rl/8B-sft-LLM评估数据集.xlsx'
OUTPUT_PATH = EXCEL_PATH.replace('.xlsx', '_with_result.xlsx')


def call_llm_model(row):
    payload = {
        "history_chat": row["最终传参上下文"],
        "user_profile": row["user_profile"],
        "rewritten_query": row["rewritten_query"],
        "history_summary": row["history_summary"],
        "user_profile_8B": row["user_profile-8B"],
        "rewritten_query_8B": row["rewritten_query-8B"],
        "history_summary_8B": row["history_summary-8B"],
        "rag_recall": row["RAG_recall_32b"],
        "rag_recall_8B": row["RAG_recall_8b"]
    }
    prompt = g_sa_prompt_manager.render_prompt(
        prompt_name="auto_eval",
        **payload,
    )
    return SafeParser.parse_json_to_dict(llm.invoke(prompt).content)

def main():
    # 1. 读Excel
    df = pd.read_excel(EXCEL_PATH)
    # 2. 遍历每行，批量推理
    new_columns = {
        "better": [],
        "reason": [],
        "score": [],
        "brief": []
    }
    for _, row in df.iterrows():
        res = call_llm_model(row)
        new_columns["better"].append(res["better"])
        new_columns["reason"].append(res["reason"])
        # 分数合并成字符串（如需拆分三列可以拆开）
        score_obj = res["score"]
        score_str = f"32b:{score_obj['32b']}, " \
                    f"8b:{score_obj['8b']}, "
        new_columns["score"].append(score_str)
        new_columns["brief"].append(res["brief"])
    # 3. 添加新列
    for key, value in new_columns.items():
        df[key] = value
    # 4. 写回新Excel
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"写入完成：{OUTPUT_PATH}")

if __name__ == '__main__':
    main()