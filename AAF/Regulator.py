
def Regulator(eval_results, adjustable_para, stop_threshold):
    loss_1 = eval_results[0]['eval_loss']
    rouge_1 = eval_results[0]['eval_rougeL']
    loss_2 = eval_results[1]['eval_loss']
    rouge_2 = eval_results[1]['eval_rougeL']

    print(f"{loss_1}, {rouge_1}")
    print(f"{loss_2}, {rouge_2}")

    Stop_symbol = 0
    # Comparison
    if loss_2 < loss_1 and rouge_2 > rouge_1:
        print("Operation: Increase diversity enhancement number")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para += 0.1
            Stop_symbol = 0

    elif loss_2 > loss_1 and rouge_2 > rouge_1:
        print("Operation: Add simple enhanced data")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            Stop_symbol = 0
    elif loss_2 < loss_1 and rouge_2 < rouge_1:
        print("Operation: Add simple enhanced data")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            Stop_symbol = 0
    elif loss_2 > loss_1 and rouge_2 < rouge_1:
        print("Reduced effectiveness and overfitting")
        if adjustable_para >= -0.3 and adjustable_para <= 0.3:
            adjustable_para -= 0.1
            stop_threshold += 1
            Stop_symbol = 1

    else:
        print("Undefined situation")

    return adjustable_para, Stop_symbol, stop_threshold


# adjustable_para, stop_train = Regulator(eval_results, adjustable_para)
