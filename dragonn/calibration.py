from abstention.calibration import PlattScaling, IsotonicRegression
import numpy as np

def get_calibration_function_classification(logits,labels):
    #need to generate separate calibration function for each task
    num_tasks=logits.shape[1]
    calibration_funcs=[]
    for task_index in range(num_tasks):
        calibration_funcs.append(PlattScaling()(
            valid_preacts=logits[:,task_index],
            valid_labels=labels.iloc[:,task_index].astype(bool)))
    return calibration_funcs

def get_calibration_function_regression(preacts,labels):
        #need to generate separate calibration function for each task
        num_tasks=preacts.shape[1]
        calibration_funcs=[]
        for task_index in range(num_tasks):
            calibration_funcs.append(IsotonicRegression()(
                valid_preacts=preacts[:,task_index],
                valid_labels=labels.iloc[:,task_index]))
        return calibration_funcs


def get_calibrated_predictions(preacts,calibration_functions):
    num_tasks=preacts.shape[1]
    num_calibration_functions=len(calibration_functions)
    assert num_tasks == num_calibration_functions
    calibrated_predictions=None
    for i in range(num_tasks):
        task_calibrated_predictions=np.expand_dims(calibration_functions[i](preacts[:,i]),axis=1)
        if calibrated_predictions is None:
            calibrated_predictions=task_calibrated_predictions
        else:
            calibrated_predictions=np.concatenate((calibrated_predictions,task_calibrated_predictions),axis=1)
    print(calibrated_predictions.shape)
    return calibrated_predictions 
