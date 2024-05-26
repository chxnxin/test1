# multi_run.py
import subprocess

def run_multiple_scripts(script_names):
    try:
        for script_name in script_names:
            subprocess.run(['python', script_name])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names
    scripts_to_run = ['run_training_KD_lam1.py', 'run_training_KD_lam1.py', 'run_training_KD_lam1.py', 'run_training_KD_lam1.py','run_training_KD_lam1.py']
    run_multiple_scripts(scripts_to_run)
