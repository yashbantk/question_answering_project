# run_project.py

import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Successfully ran {script_name}:\n{result.stdout}")

if __name__ == "__main__":
    scripts = [
        'scripts/data_preprocessing.py',
        'scripts/train_model.py',
        'scripts/evaluate_model.py',
        'scripts/visualize_results.py'
    ]

    for script in scripts:
        run_script(script)
