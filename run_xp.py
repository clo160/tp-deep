import os

# Define the values of eps
## To do 14
eps_values = []

# Loop through each value and execute the script
for eps in eps_values:
    print(f"Running evaluate.py with eps={eps}")
    command = f"python evaluate.py --path base --model cnn --attack fgsm --eps {eps}"
    os.system(command)
