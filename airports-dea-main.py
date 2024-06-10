import csv
import numpy as np
import pulp
import pandas as pd
from pathlib import Path

def read_airport_names_from_csv(file_path, delimiter=';'):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader)
        for row in reader:
            data.append(row[0])
    return np.array(data)

def read_values_from_csv(file_path, delimiter=';'):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader)
        for row in reader:
            row = [float(value) for index, value in enumerate(row) if index != 0]
            data.append(row)
    return np.array(data)

def read_weight_samples_from_csv(file_path, delimiter=';', num_inputs=4, num_outputs=2):
    input_weights = []
    output_weights = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader)
        for row in reader:
            row = [float(value) for value in row]
            input_weights.append(row[1:num_inputs+1])
            output_weights.append(row[num_inputs+1:])
    return np.array(input_weights), np.array(output_weights)

def construct_CCR_problem(inputs, outputs, examined_instance):
    problem = pulp.LpProblem('CCR', pulp.LpMinimize)

    theta = pulp.LpVariable('theta', lowBound=0, cat='Continuous')

    problem += theta, 'theta'

    num_instances = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    lambda_vars = [pulp.LpVariable(f'lambda_{i}', lowBound=0, cat='Continuous') for i in range(num_instances)]
    
    for i in range(num_inputs):
        problem += pulp.lpSum(lambda_vars[j] * inputs[j, i] for j in range(num_instances)) <= theta * inputs[examined_instance, i], f'input_{i}_constraint'

    for i in range(num_outputs):
        problem += pulp.lpSum(lambda_vars[j] * outputs[j, i] for j in range(num_instances)) >= outputs[examined_instance, i], f'output_{i}_constraint'

    return problem, lambda_vars

def construct_CCR_super_efficiency_problem(inputs, outputs, examined_instance):
    problem = pulp.LpProblem('CCR_super_efficiency', pulp.LpMaximize)

    num_instances = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    input_weights = [pulp.LpVariable(f'input_weight_{i}', lowBound=0, cat='Continuous') for i in range(num_inputs)]
    output_weights = [pulp.LpVariable(f'output_weight_{i}', lowBound=0, cat='Continuous') for i in range(num_outputs)]

    problem += pulp.lpSum([output_weights[i] * outputs[examined_instance, i] for i in range(num_outputs)]), 'super_efficiency'

    problem += pulp.lpSum([input_weights[i] *  inputs[examined_instance, i] for i in range(num_inputs)]) == 1, 'input_sum_constraint'

    for i in range(num_instances):
        if i == examined_instance:
            continue
        problem += pulp.lpSum([output_weights[j] * outputs[i, j] for j in range(num_outputs)]) <= pulp.lpSum([input_weights[j] * inputs[i, j] for j in range(num_inputs)]), f'instance_{i}_constraint'

    return problem

def construct_CCR_efficienncy_for_CE(inputs, outputs, examined_instance):
    problem = pulp.LpProblem('CCR_eff_ce', pulp.LpMaximize)

    num_instances = inputs.shape[0]
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]

    input_weights = [pulp.LpVariable(f'input_weight_{i}', lowBound=0, cat='Continuous') for i in range(num_inputs)]
    output_weights = [pulp.LpVariable(f'output_weight_{i}', lowBound=0, cat='Continuous') for i in range(num_outputs)]

    problem += pulp.lpSum([output_weights[i] * outputs[examined_instance, i] for i in range(num_outputs)]), 'super_efficiency'

    problem += pulp.lpSum([input_weights[i] *  inputs[examined_instance, i] for i in range(num_inputs)]) == 1, 'input_sum_constraint'

    for i in range(num_instances):
        problem += pulp.lpSum([output_weights[j] * outputs[i, j] for j in range(num_outputs)]) <= pulp.lpSum([input_weights[j] * inputs[i, j] for j in range(num_inputs)]), f'instance_{i}_constraint'

    return problem, input_weights, output_weights

def find_CCR_efficiency_and_hcu(inputs, outputs, examined_instance):
    problem, lambda_vars = construct_CCR_problem(inputs, outputs, examined_instance)
    problem.solve()
    efficiency = pulp.value(problem.objective)
    lambdas = np.array([pulp.value(lambda_var) for lambda_var in lambda_vars])[:, np.newaxis]
    hcu_inputs = np.sum(lambdas * inputs, axis=0)
    hcu_outputs = np.sum(lambdas * outputs, axis=0)
    return efficiency, hcu_inputs, hcu_outputs

def find_CCR_super_efficiency(inputs, outputs, examined_instance):
    problem = construct_CCR_super_efficiency_problem(inputs, outputs, examined_instance)
    problem.solve()
    return pulp.value(problem.objective)

def find_CCR_efficiency_and_weights(inputs, outputs, examined_instance):
    problem, input_weights, output_weights = construct_CCR_efficienncy_for_CE(inputs, outputs, examined_instance)
    problem.solve()
    efficiency = pulp.value(problem.objective)
    input_weights = np.array([pulp.value(input_weight) for input_weight in input_weights])
    output_weights = np.array([pulp.value(output_weight) for output_weight in output_weights])
    return efficiency, input_weights, output_weights

def aiport_effeciency_analysis(inputs, outputs, airport_names):
    results = []
    for i in range(inputs.shape[0]):
        efficiency, hcu_inputs, hcu_outputs = find_CCR_efficiency_and_hcu(inputs, outputs, i)
        result = [airport_names[i], efficiency]
        result += list(hcu_inputs)
        result += list(hcu_outputs)
        result += list(inputs[i] - hcu_inputs)
        result += list(outputs[i] - hcu_outputs)
        results.append(result)

    column_names = ['Airport', 'Efficiency']
    column_names += [f"HCU i_{i}" for i in range(inputs.shape[1])]
    column_names += [f"HCU o_{i}" for i in range(outputs.shape[1])]
    column_names += [f'HCU_diff i_{i}' for i in range(inputs.shape[1])]
    column_names += [f'HCU_diff o_{i}' for i in range(outputs.shape[1])]

    return pd.DataFrame(results, columns=column_names)

def airport_super_efficiency_analysis(inputs, outputs, airport_names):
    results = []
    for i in range(inputs.shape[0]):
        super_efficiency = find_CCR_super_efficiency(inputs, outputs, i)
        results.append([airport_names[i], super_efficiency])

    return pd.DataFrame(results, columns=['Airport', 'Super Efficiency'])

def efficiency_from_weights(input, output, input_weights, output_weights):
    return np.sum(output_weights * output) / np.sum(input_weights * input)

def airport_cross_efficiency_analysis(inputs, outputs, airport_names):
    num_instances = inputs.shape[0]

    input_weights = []
    output_weights = []
    for i in range(num_instances):
        efficiency, input_weight, output_weight = find_CCR_efficiency_and_weights(inputs, outputs, i)
        input_weights.append(input_weight)
        output_weights.append(output_weight)

    results = []

    for i in range(num_instances):
        ce_results = [efficiency_from_weights(inputs[i, :], outputs[i, :], input_weights[j], output_weights[j]) for j in range(num_instances)]
        avg_ce = np.mean(ce_results)
        results.append([airport_names[i]] + ce_results + [avg_ce])

    column_names = ['Airport']
    column_names += [f'{airport_names[i]}' for i in range(num_instances)]
    column_names += ['Avg']

    return pd.DataFrame(results, columns=column_names)

def airport_efficiency_distribution_analysis(inputs, outputs, input_weights, output_weights, airport_names):
    num_instances = inputs.shape[0]
    num_weight_samples = input_weights.shape[0]

    efficiences = np.zeros((num_instances, num_weight_samples))

    for i in range(num_instances):
        for j in range(num_weight_samples):
            efficiences[i, j] = efficiency_from_weights(inputs[i, :], outputs[i, :], input_weights[j], output_weights[j])

    efficiences = efficiences / np.max(efficiences)

    distribution_ranges = np.linspace(0, 1, 6)

    results = []
    for i in range(num_instances):
        distribution = np.histogram(efficiences[i, :], bins=distribution_ranges)[0]
        distribution = distribution / np.sum(distribution)
        results.append([airport_names[i]] + list(distribution) + [np.mean(efficiences[i, :])])

    column_names = ['Airport']
    column_names += [f'{distribution_ranges[i]}-{distribution_ranges[i+1]}' for i in range(len(distribution_ranges) - 1)]
    column_names += ['EE']

    return pd.DataFrame(results, columns=column_names)

def main():
    Path("out").mkdir(parents=True, exist_ok=True)

    inputs_path = 'data/inputs.csv'
    outputs_path = 'data/outputs.csv'
    weight_samples_path = 'data/samples_homework.csv'
    inputs = read_values_from_csv(inputs_path)
    outputs = read_values_from_csv(outputs_path)
    input_weights, output_weights = read_weight_samples_from_csv(weight_samples_path, num_inputs=inputs.shape[1], num_outputs=outputs.shape[1])
    airport_names = read_airport_names_from_csv(inputs_path)
    
    results = aiport_effeciency_analysis(inputs, outputs, airport_names)

    efficiency_results = results[['Airport', 'Efficiency']]
    efficiency_results.to_csv('out/efficiency.csv', float_format='%.3f', index=False)

    only_inputs = results[['Airport'] + [f'HCU i_{i}' for i in range(inputs.shape[1])] + [f'HCU_diff i_{i}' for i in range(inputs.shape[1])]][results['Efficiency'] < 1.0]
    only_inputs.to_csv('out/HCU_inputs.csv', float_format='%.3f', index=False)

    super_efficiency_results = airport_super_efficiency_analysis(inputs, outputs, airport_names)
    super_efficiency_results.to_csv('out/super_efficiency.csv', float_format='%.3f', index=False)

    cross_efficiency_results = airport_cross_efficiency_analysis(inputs, outputs, airport_names)
    cross_efficiency_results.to_csv('out/cross_efficiency.csv', float_format='%.3f', index=False)

    efficiency_distribution_results = airport_efficiency_distribution_analysis(inputs, outputs, input_weights, output_weights, airport_names)
    efficiency_distribution_results.to_csv('out/efficiency_distribution.csv', float_format='%.3f', index=False)

    print("Super-efficency ranking:")
    print(super_efficiency_results.sort_values(by='Super Efficiency', ascending=False))

    print("Cross-efficiency ranking:")
    print(cross_efficiency_results.sort_values(by='Avg', ascending=False))

    print("Efficiency distribution ranking:")
    print(efficiency_distribution_results.sort_values(by='EE', ascending=False))
    

main()