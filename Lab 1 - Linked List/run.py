import subprocess
import statistics
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
import os

NO_OF_SAMPLES = 500  # Number of samples used

# Compiling the source codes in C
def compileAll():
    subprocess.call(['gcc', '-g', '-Wall', '-o', 'serial_program.exe', 'serial_program.c'])
    subprocess.call(['gcc', '-g', '-Wall', '-o', 'parallel_mutex_program.exe', 'parallel_mutex_program.c', '-lm', '-lpthread'])
    subprocess.call(['gcc', '-g', '-Wall', '-o', 'parallel_rw_lock_program.exe', 'parallel_rw_lock_program.c', '-lm', '-lpthread'])

# Execution of given process and calculation of average and standard deviation
def execute(command):
    elapsed_times = []

    for _ in range(NO_OF_SAMPLES):
        time = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
        elapsed_times.append(float(time))

    avg = statistics.mean(elapsed_times)
    standard_deviation = statistics.stdev(elapsed_times)
    samples = math.ceil(((100 * 1.96 * standard_deviation) / (5 * avg)) ** 2)

    print(f'Average: {avg:.5f}')
    print(f'Standard Deviation: {standard_deviation:.5f}')
    print(f'Samples: {samples}')

    return avg, standard_deviation, samples

# Execution of a list of commands
def executeCommands(cmds):
    threads = []
    averages = []
    standard_deviations = []
    samples_list = []
    for i in range(len(cmds)):
        num_threads = 2**i
        print(f"Number of Threads: {num_threads}")
        avg, std_dev, samples = execute(cmds[i])
        averages.append(avg)
        standard_deviations.append(std_dev)
        samples_list.append(samples)
        threads.append(num_threads)
        print("")
    return threads, averages, standard_deviations, samples_list

# Plotting the results and saving the plot as an image
def plot_results(case, threads, serial_avg, mutex_averages, rw_averages, filename):
    plt.plot(threads, [serial_avg] * len(threads), marker='o', label='Serial')
    plt.plot(threads, mutex_averages, marker='o', label='Mutex')
    plt.plot(threads, rw_averages, marker='o', label='Read-Write')
    
    plt.title(f'Case {case}: Execution Time vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Execution Time (ms)')
    plt.xscale('linear')  # Linear scale for threads
    plt.yscale('linear')  # Linear scale for time in milliseconds
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Save table data to a CSV file
def save_table_to_csv(table_data, filename):
    with open(os.path.join(output_dir, filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table_data)

# Compile all the files
compileAll()

# Commands to be executed
serial = [['./serial_program.exe', '1000', '10000', '0.99', '0.005', '0.005'], 
          ['./serial_program.exe', '1000', '10000', '0.9', '0.05', '0.05'], 
          ['./serial_program.exe', '1000', '10000', '0.5', '0.25', '0.25']]

mutex_1 = [['./parallel_mutex_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '1'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '2'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '4'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '8']]

mutex_2 = [['./parallel_mutex_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '1'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '2'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '4'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '8']]

mutex_3 = [['./parallel_mutex_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '1'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '2'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '4'], 
           ['./parallel_mutex_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '8']]

rw_1 = [['./parallel_rw_lock_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '1'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '2'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '4'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.99', '0.005', '0.005', '8']]

rw_2 = [['./parallel_rw_lock_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '1'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '2'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '4'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.9', '0.05', '0.05', '8']]

rw_3 = [['./parallel_rw_lock_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '1'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '2'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '4'], 
        ['./parallel_rw_lock_program.exe', '1000', '10000', '0.5', '0.25', '0.25', '8']]

mutex = [mutex_1, mutex_2, mutex_3]
rw = [rw_1, rw_2, rw_3]

# Ensure the directory exists
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Execute and plot the output
for i in range(1, 4):
    print(f'=============== CASE: {i} ===============')

    print('Serial linked list ')
    print('=======')
    serial_avg, serial_std_dev, serial_samples = execute(serial[i-1])
    print('')

    print('Mutex linked list ')
    print('=======')
    threads, mutex_averages, mutex_standard_deviations, mutex_samples = executeCommands(mutex[i-1])
    print('')

    print('Read-Write linked list ')
    print('=======')
    _, rw_averages, rw_standard_deviations, rw_samples = executeCommands(rw[i-1])
    
    # Save the results for this case
    plot_results(i, threads, serial_avg, mutex_averages, rw_averages, f'case_{i}_comparison.png')

    # Generate table of results
    table_data = [
        ["Threads", "1", "1", "2", "2", "4", "4", "8", "8"],
        ["Method", "Avg", "Std", "Avg", "Std", "Avg", "Std", "Avg", "Std"],
        ["Serial", f"{serial_avg:.5f}", f"{serial_std_dev:.5f}", "-", "-", "-", "-", "-", "-"],
        ["Mutex", f"{mutex_averages[0]:.5f}", f"{mutex_standard_deviations[0]:.5f}", f"{mutex_averages[1]:.5f}", f"{mutex_standard_deviations[1]:.5f}", f"{mutex_averages[2]:.5f}", f"{mutex_standard_deviations[2]:.5f}", f"{mutex_averages[3]:.5f}", f"{mutex_standard_deviations[3]:.5f}"],
        ["Read-Write", f"{rw_averages[0]:.5f}", f"{rw_standard_deviations[0]:.5f}", f"{rw_averages[1]:.5f}", f"{rw_standard_deviations[1]:.5f}", f"{rw_averages[2]:.5f}", f"{rw_standard_deviations[2]:.5f}", f"{rw_averages[3]:.5f}", f"{rw_standard_deviations[3]:.5f}"]
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    save_table_to_csv(table_data, f'case_{i}_results.csv')
    print('')

    # Generate table of samples
    samples_table_data = [
        ["Threads", "1", "2", "4", "8"],
        ["Serial", serial_samples, "-", "-", "-"],
        ["Mutex", mutex_samples[0], mutex_samples[1], mutex_samples[2], mutex_samples[3]],
        ["Read-Write", rw_samples[0], rw_samples[1], rw_samples[2], rw_samples[3]]
    ]
    print(tabulate(samples_table_data, headers="firstrow", tablefmt="grid"))
    save_table_to_csv(samples_table_data, f'case_{i}_samples.csv')
    print('')