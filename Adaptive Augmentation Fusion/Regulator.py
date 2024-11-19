
def Regulator(eval_results, adjustable_para, stop_threshold):
    loss_1 = eval_results[0]['eval_loss']
    rouge_1 = eval_results[0]['eval_rougeL']
    loss_2 = eval_results[1]['eval_loss']
    rouge_2 = eval_results[1]['eval_rougeL']

    print(f"{loss_1}, {rouge_1}")
    print(f"{loss_2}, {rouge_2}")

    Stop_symbol = 0
    # 比较结果
    if loss_2 < loss_1 and rouge_2 > rouge_1:
        print("操作：增加多样性增强数据")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para += 0.1
            Stop_symbol = 0

    elif loss_2 > loss_1 and rouge_2 > rouge_1:
        print("操作：增加简单增强数据")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            Stop_symbol = 0
    elif loss_2 < loss_1 and rouge_2 < rouge_1:
        print("操作：增加简单增强数据")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            Stop_symbol = 0
    elif loss_2 > loss_1 and rouge_2 < rouge_1:
        print("效果降低, 过拟合")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            stop_threshold += 1
            Stop_symbol = 1

    else:
        print("未定义的情况")

    return adjustable_para, Stop_symbol, stop_threshold


# adjustable_para, stop_train = Regulator(eval_results, adjustable_para)
