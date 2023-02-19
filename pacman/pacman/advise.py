def determine_give_advice(
    advice_budget,
    advice_strategy,
    state_importance,
    teacher_action,
    student_action,
    call_counter,
    eval=False
):
    if advice_budget == 0:
        return False, 0

    if advice_budget > 0:
        if advice_strategy == 'e':
            if not eval:
                advice_budget -= 1
            # if advice_budget == 0:
            #     print ('advice budget used up!!!!!!!!!!!!!!!!')
            return True, advice_budget

        elif advice_strategy == 'a':
            alternative_advice_freq = 5
            if call_counter % alternative_advice_freq == 0:
                if not eval:
                    advice_budget -= 1
                # if advice_budget == 0:
                #     print ('advice budget used up!!!!!!!!!!!!!!!!')
                return True, advice_budget
            else:
                return False, advice_budget

        elif advice_strategy == 'i':
            state_importance_threshold = 0.4
            if state_importance >= state_importance_threshold:
                if not eval:
                    advice_budget -= 1
                # if advice_budget == 0:
                #     print ('advice budget used up!!!!!!!!!!!!!!!!')
                return True, advice_budget
            else:
                return False, advice_budget

        elif advice_strategy == 'm':
            state_importance_threshold = 0.3
            if state_importance >= state_importance_threshold and teacher_action != student_action:
                if not eval:
                    advice_budget -= 1
                # if advice_budget == 0:
                #     print ('advice budget used up!!!!!!!!!!!!!!!!')
                return True, advice_budget
            else:
                return False, advice_budget
        else:
            print ('unknown advice strategy!')
            return False, advice_budget
