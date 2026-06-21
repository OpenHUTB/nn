# 训练日志记录工具
def record_train_data(train_round, avg_reward):
    with open("train_record.txt", "a", encoding="utf-8") as f:
        f.write(f"训练轮数：{train_round}  平均奖励值：{avg_reward}\n")

def clear_log():
    with open("train_record.txt","w",encoding="utf-8") as f:
        f.write("")
    print("日志已清空")